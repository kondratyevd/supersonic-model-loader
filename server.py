from utils import format_model_label, escape_model_name, find_free_port
from logger import get_logger
from kubernetes import client
import grpc
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferenceServerClient
from typing import Dict
import atexit
import subprocess
import os
import signal
import time
import queue
import threading
import requests

class Server:
    def __init__(self, pod):
        self.logger = get_logger("server")
        self.pod = pod
        self.pod_name = pod.metadata.name
        self.pod_namespace = pod.metadata.namespace
        self.v1 = client.CoreV1Api()

        self.port_forward_process = None
        self.port_forward_local_port = None
        self.port_forward_remote_port = None

        atexit.register(self.cleanup_port_forward)

        # Get release name from pod labels
        self.release_name = self.pod.metadata.labels.get("app.kubernetes.io/instance", "supersonic-test")

    def refresh_pod(self):
        """
        Refresh the pod state and update class properties.
        This ensures we have the most up-to-date pod information.
        """
        try:
            self.pod = self.v1.read_namespaced_pod(
                name=self.pod_name,
                namespace=self.pod_namespace
            )
            self.logger.debug("Refreshed pod state", 
                            pod=self.pod_name)
        except Exception as e:
            self.logger.error("Failed to refresh pod state",
                            error=str(e),
                            pod=self.pod_name)
            raise

    def restart_pod(self):
        """
        Restart the Triton server pod by deleting it and letting the controller create a new one.
        This will trigger a pod restart while maintaining the same pod name.
        """
        try:
            self.logger.info("Initiating pod restart", pod=self.pod_name)
            
            # Delete the pod
            self.v1.delete_namespaced_pod(
                name=self.pod_name,
                namespace=self.pod_namespace
            )
            
            # Wait for the pod to be recreated and become ready
            while True:
                try:
                    self.refresh_pod()
                    if self.pod.status.phase == "Running":
                        self.logger.info("Pod successfully restarted", pod=self.pod_name)
                        break
                except Exception as e:
                    self.logger.debug("Waiting for pod to be recreated", 
                                    error=str(e),
                                    pod=self.pod_name)
                time.sleep(1)
                
        except Exception as e:
            self.logger.error("Failed to restart pod", 
                            error=str(e),
                            pod=self.pod_name)
            raise

    def setup_port_forward(self, remote_port: int) -> int:
        """
        Set up port-forwarding to the server pod.
        """
        port_queue = queue.Queue()
        
        def port_forward():
            try:
                # Find a free port
                free_port = find_free_port()
                
                cmd = [
                    "kubectl", "port-forward",
                    f"pod/{self.pod_name}",
                    f"{free_port}:{remote_port}",
                    "-n", self.pod_namespace
                ]
                
                self.port_forward_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
                
                time.sleep(2)
                
                if self.port_forward_process.poll() is not None:
                    stderr = self.port_forward_process.stderr.read().decode()
                    self.logger.error("Port-forward failed", 
                                    error=stderr,
                                    pod=self.pod_name)
                    raise
                
                port_queue.put(free_port)
                
                while True:
                    if self.port_forward_process.poll() is not None:
                        stderr = self.port_forward_process.stderr.read().decode()
                        self.logger.error("Port-forward terminated", 
                                        error=stderr,
                                        pod=self.pod_name)
                        raise
                    time.sleep(0.1)
            except Exception as e:
                port_queue.put(None)
        
        pf_thread = threading.Thread(target=port_forward, daemon=True)
        pf_thread.start()
        
        local_port = port_queue.get()
        if local_port is None:
            raise Exception("Failed to establish port-forwarding")
        
        self.logger.debug("Port-forwarding established", 
                       pod=self.pod_name,
                       local_port=local_port,
                       remote_port=remote_port)
        
        self.port_forward_local_port = local_port
        self.port_forward_remote_port = remote_port
        return local_port

    def cleanup_port_forward(self):
        """Clean up port-forward process if it exists"""
        if self.port_forward_process:
            try:
                os.killpg(os.getpgid(self.port_forward_process.pid), signal.SIGTERM)
                self.logger.debug("Port-forwarding cleaned up",
                                 local_port=self.port_forward_local_port,
                                 remote_port=self.port_forward_remote_port)
            except:
                pass
            self.port_forward_process = None
            self.port_forward_local_port = None
            self.port_forward_remote_port = None

    def get_triton_client(self, port: int) -> InferenceServerClient:
        """
        Create a Triton gRPC client.
        """
        try:
            client = grpcclient.InferenceServerClient(
                url=f"localhost:{port}",
                verbose=False
            )
            return client
        except Exception as e:
            self.logger.error("Failed to create Triton client",
                            error=str(e),
                            pod=self.pod_name)
            raise

    def get_models(self) -> Dict[str, Dict]:
        """
        Get the models loaded on the Triton Inference Server API.
        Uses kubectl port-forward for direct pod access.
        """
        self.logger.debug("Querying Triton server for loaded models", 
                        pod=self.pod_name)
        
        try:
            local_port = self.setup_port_forward(8001)  # gRPC port
            client = self.get_triton_client(local_port)
            
            # Get repository index
            repository_index = client.get_model_repository_index()
            
            ready_models_count = 0

            for model in repository_index.models:
                model_name = model.name
                model_version = model.version
                model_name_full = f"{model_name}-v{model_version}"
                
                # First check if model is ready
                is_ready = client.is_model_ready(model_name)
                
                if is_ready:
                    if not self.has_label(model_name_full):
                        self.logger.debug("Model is ready, adding label to pod", 
                                    model=model_name,
                                    version=model_version,
                                    state=model.state,
                                    pod=self.pod_name)
                        self.add_label(model_name_full)
                    ready_models_count += 1
                elif model.version == "":
                    self.logger.debug("Model is in repository but no versions are loaded to the server", 
                                   model=model_name,
                                   pod=self.pod_name)
                else:
                    if self.has_label(model_name_full):
                        self.logger.warning("Model is in repository but not ready, removing label", 
                                    model=model_name,
                                    version=model_version,
                                    pod=self.pod_name)
                        self.remove_label(model_name_full)
            
            self.logger.info("Model information retrieved", 
                           pod=self.pod_name,
                           ready_model_count=ready_models_count,
                           total_model_count=len(repository_index.models))
            
        except Exception as e:
            self.logger.error("Failed to query Triton server", 
                            error=str(e),
                            pod=self.pod_name)
            raise
        finally:
            self.cleanup_port_forward()

    def count_versions(self, model_name: str, client: InferenceServerClient, state: str = None) -> list:
        """
        Count versions of a model in the repository.
        """
        repository_index = client.get_model_repository_index()
        versions = []
        
        for model in repository_index.models:
            if model.name == model_name and (state is None or model.state == state):
                if model.version != '':
                    versions.append(model.version)
        
        return versions

    def load_or_unload_model(self, model_name: str, load: bool):
        """
        Load or unload a model into the Triton server
        """
        action = "load" if load else "unload"
        try:
            local_port = self.setup_port_forward(8001)  # gRPC port
            client = self.get_triton_client(local_port)
            
            # Get all versions from repository
            model_versions = self.count_versions(model_name, client)
            model_versions_ready = self.count_versions(model_name, client, state="READY")

            self.logger.info(f"Counting model versions",
                        model=model_name,
                        versions_in_repo=len(model_versions),
                        versions_ready=len(model_versions_ready),
                        pod=self.pod_name)
            
            if load:
                # First load the model, then add labels for loaded versions
                client.load_model(model_name)
                
                loaded_versions = self.count_versions(model_name, client, state="READY")
                
                for version in loaded_versions:
                    self.add_label(f"{model_name}-v{version}")
                
                self.logger.info(f"Model loaded successfully", 
                               model=model_name,
                               version_count=len(loaded_versions),
                               pod=self.pod_name)
            else:
                # First remove labels for all versions, then unload the model
                for version in model_versions:
                    self.remove_label(f"{model_name}-v{version}")
                client.unload_model(model_name)
                
                self.logger.info(f"Model unloaded successfully", 
                               model=model_name,
                               pod=self.pod_name)
            
        except Exception as e:
            self.logger.error(f"Failed to {action} model", 
                            model=model_name,
                            error=str(e),
                            pod=self.pod_name)
            raise
        finally:
            self.cleanup_port_forward()

    def load_model(self, model_name: str):
        self.load_or_unload_model(model_name, load=True)

    def unload_model(self, model_name: str):
        self.load_or_unload_model(model_name, load=False)

    def has_label(self, model_name: str) -> bool:
        """
        Check if a model label exists on the pod.
        """
        label_key = format_model_label(model_name)
        
        try:
            self.refresh_pod()
            
            has_label = (
                self.pod.metadata.labels is not None and 
                label_key in self.pod.metadata.labels and 
                self.pod.metadata.labels[label_key] == "true"
            )
            
            self.logger.debug("Checked if label exists", 
                            label=label_key,
                            exists=has_label,
                            pod=self.pod_name)
            
            return has_label
            
        except Exception as e:
            self.logger.error("Failed to check model label", 
                            model=model_name,
                            error=str(e),
                            pod=self.pod_name)
            raise

    def add_label(self, model_name: str):
        """
        Add a label to the Triton server indicating a model is loaded
        Format: sonic.model.loaded/modelname-v1: true
        """
        label_key = format_model_label(model_name)
        
        try:
            # Check if label already exists
            if self.has_label(model_name):
                self.logger.debug("Model label already exists", 
                                  label=label_key,
                                  pod=self.pod_name)
                return
            
            # Add the new label
            if self.pod.metadata.labels is None:
                self.pod.metadata.labels = {}
            
            self.pod.metadata.labels[label_key] = "true"
            
            # Update the pod
            self.v1.patch_namespaced_pod(
                name=self.pod_name,
                namespace=self.pod_namespace,
                body={"metadata": {"labels": self.pod.metadata.labels}}
            )
            self.logger.info("Added model label to pod", 
                            label=label_key,
                            pod=self.pod_name)
            
        except Exception as e:
            self.logger.error("Failed to add model label", 
                            model=model_name,
                            error=str(e),
                            pod=self.pod_name)
            raise

    def remove_label(self, model_name: str):
        """
        Remove a label from the Triton server
        """
        label_key = format_model_label(model_name)
        
        try:
            if not self.has_label(model_name):
                self.logger.debug("Model label not found", 
                                  label=label_key,
                                  pod=self.pod_name)
                return
                
            # Create a new labels dict and set the label value to None
            new_labels = dict(self.pod.metadata.labels)
            new_labels[label_key] = None
            
            # Update the pod with new labels
            self.v1.patch_namespaced_pod(
                name=self.pod_name,
                namespace=self.pod_namespace,
                body={"metadata": {"labels": new_labels}}
            )
            self.logger.info("Removed model label from pod", 
                            label=label_key,
                            pod=self.pod_name)
                
        except Exception as e:
            self.logger.error("Failed to remove model label", 
                            model=model_name,
                            error=str(e),
                            pod=self.pod_name)
            raise

    def get_gpu_memory(self) -> Dict[str, Dict[str, int]]:
        """
        Get GPU memory information from the Triton server.
        """
        self.logger.info("Querying GPU memory information", 
                        pod=self.pod_name)
        
        try:
            local_port = self.setup_port_forward(8002)  # metrics port
            
            # Get metrics from the HTTP endpoint
            response = requests.get(f"http://localhost:{local_port}/metrics")
            response.raise_for_status()
            metrics_text = response.text
            
            gpu_memory = {}
            
            # Process GPU metrics
            for line in metrics_text.split('\n'):
                if not line or line.startswith('#'):
                    continue
                    
                # Parse Prometheus metric line
                # Format: metric_name{label1="value1",label2="value2"} value
                try:
                    metric_name, rest = line.split('{', 1)
                    labels_str, value = rest.split('}', 1)
                    value = float(value.strip())
                    
                    # Parse labels
                    labels = {}
                    for label in labels_str.split(','):
                        if '=' in label:
                            key, val = label.split('=', 1)
                            labels[key] = val.strip('"')
                    
                    gpu_uuid = labels.get('gpu_uuid')
                    if gpu_uuid is None:
                        continue
                        
                    if gpu_uuid not in gpu_memory:
                        gpu_memory[gpu_uuid] = {}
                    
                    if metric_name == "nv_gpu_memory_total_bytes":
                        gpu_memory[gpu_uuid]['total_memory'] = int(value)
                    elif metric_name == "nv_gpu_memory_used_bytes":
                        gpu_memory[gpu_uuid]['used_memory'] = int(value)
                    elif metric_name in ['nv_gpu_utilization', 'nv_gpu_power_usage', 'nv_gpu_temperature']:
                        self.logger.debug(f"GPU {metric_name}", 
                                       gpu_uuid=gpu_uuid,
                                       value=value,
                                       pod=self.pod_name)
                except Exception as e:
                    self.logger.warning(f"Failed to parse metric line: {line}", 
                                      error=str(e),
                                      pod=self.pod_name)
                    continue
            
            # Calculate free memory for each GPU
            for gpu_uuid in gpu_memory:
                if 'total_memory' in gpu_memory[gpu_uuid] and 'used_memory' in gpu_memory[gpu_uuid]:
                    gpu_memory[gpu_uuid]['free_memory'] = (
                        gpu_memory[gpu_uuid]['total_memory'] - 
                        gpu_memory[gpu_uuid]['used_memory']
                    )
                    
                    # Log memory information for each GPU
                    self.logger.info("GPU memory information", 
                                   gpu_uuid=gpu_uuid,
                                   total_mb=gpu_memory[gpu_uuid]['total_memory'] // (1024 * 1024),
                                   used_mb=gpu_memory[gpu_uuid]['used_memory'] // (1024 * 1024),
                                   free_mb=gpu_memory[gpu_uuid]['free_memory'] // (1024 * 1024),
                                   pod=self.pod_name)
            
            return gpu_memory
            
        except Exception as e:
            self.logger.error("Failed to query Triton metrics", 
                            error=str(e),
                            pod=self.pod_name)
            raise
        finally:
            self.cleanup_port_forward()
