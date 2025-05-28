from utils import format_model_label, escape_model_name, find_free_port
from logger import get_logger
from kubernetes import client
import requests
from typing import Dict
import atexit
import json
import subprocess
import os
import signal
import time
import queue
import threading

class Server:
    def __init__(self, pod):
        self.logger = get_logger("server")
        self.pod = pod
        self.pod_name = pod.metadata.name
        self.pod_namespace = pod.metadata.namespace
        self.v1 = client.CoreV1Api()
        self.port_forward_process = None
        atexit.register(self.cleanup_port_forward)
        self.triton_models_info = {}
        # Get release name from pod labels
        self.release_name = self.pod.metadata.labels.get("app.kubernetes.io/instance", "supersonic-test")

    def setup_port_forward(self, remote_port: int) -> int:
        """
        Set up port-forwarding to the pod.
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
                    raise Exception(f"Port-forward failed: {stderr}")
                
                port_queue.put((free_port, self.port_forward_process))
                
                while True:
                    if self.port_forward_process.poll() is not None:
                        stderr = self.port_forward_process.stderr.read().decode()
                        raise Exception(f"Port-forward terminated: {stderr}")
                    time.sleep(0.1)
            except Exception as e:
                self.logger.warning("Port-forwarding terminated", 
                                    error=str(e),
                                    pod=self.pod_name)
                port_queue.put(None)
        
        pf_thread = threading.Thread(target=port_forward, daemon=True)
        pf_thread.start()
        
        result = port_queue.get()
        if result is None:
            raise Exception("Failed to establish port-forwarding")
        
        local_port, process = result
        self.logger.info("Port-forwarding established", 
                       pod=self.pod_name,
                       local_port=local_port)
        
        return local_port

    def cleanup_port_forward(self):
        """Clean up port-forward process if it exists"""
        if self.port_forward_process:
            try:
                os.killpg(os.getpgid(self.port_forward_process.pid), signal.SIGTERM)
                self.logger.info("Port-forwarding cleaned up")
            except:
                pass
            self.port_forward_process = None

    def query_triton_server(self, url: str, method: str = "POST") -> requests.Response:
        """
        Query the Triton server API.
        
        Args:
            url: The URL to query
            method: HTTP method to use ("GET" or "POST")
        """
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        if method == "GET":
            response = requests.get(url, headers=headers)
        else:
            response = requests.post(url, headers=headers)
            
        response.raise_for_status()
        return response

    def get_models(self) -> Dict[str, Dict]:
        """
        Get the models loaded on the Triton Inference Server API.
        Uses kubectl port-forward for direct pod access.
        """
        self.logger.info("Querying Triton server for loaded models", 
                        pod=self.pod_name)
        
        try:
            local_port = self.setup_port_forward(8000)
            
            base_url = f"http://localhost:{local_port}"
            
            response = self.query_triton_server(f"{base_url}/v2/repository/index")
            
            try:
                models_data = response.json()
            except json.JSONDecodeError as e:
                self.logger.error("Invalid JSON response from repository index", 
                                error=str(e),
                                content=response.text)
                raise
            
            for model_data in models_data:
                if not isinstance(model_data, dict):
                    self.logger.warning("Skipping invalid model data", 
                                      model_data=model_data)
                    continue
                
                model_name = model_data.get('name')
                if not model_name:
                    self.logger.warning("Skipping model with no name", 
                                      model_data=model_data)
                    continue
                
                status_response = self.query_triton_server(
                    f"{base_url}/v2/models/{model_name}/ready",
                    method="GET"
                )
                is_ready = status_response.text.lower() == 'true'
                
                config_response = self.query_triton_server(
                    f"{base_url}/v2/models/{model_name}/config",
                    method="GET"
                )
                
                try:
                    config_data = config_response.json()
                except json.JSONDecodeError as e:
                    self.logger.error("Invalid JSON response from model config endpoint", 
                                    model=model_name,
                                    error=str(e),
                                    content=config_response.text)
                    raise
                
                self.triton_models_info[model_name] = {
                    "name": model_name,
                    "version": model_data.get("version", config_data.get("version", "unknown")),
                    "state": model_data.get("state", "READY" if is_ready else "UNAVAILABLE"),
                }
                
                self.logger.info("Model information retrieved", 
                               model=model_name,
                               version=self.triton_models_info[model_name]["version"],
                               state=self.triton_models_info[model_name]["state"],
                               pod=self.pod_name)
            
        except requests.exceptions.RequestException as e:
            self.logger.error("Failed to query Triton server API", 
                            error=str(e),
                            pod=self.pod_name)
            raise
        except Exception as e:
            self.logger.error("Unexpected error while getting models", 
                            error=str(e),
                            pod=self.pod_name)
            raise
        finally:
            self.cleanup_port_forward()
    
    def sync_labels(self):
        """
        Sync the labels on the Triton server with the labels on the local pod
        """
        self.logger.info("Starting label sync for models", 
                        pod=self.pod_name,
                        model_count=len(self.triton_models_info))
        
        for model_name, model_info in self.triton_models_info.items():
            model_name = escape_model_name(model_info["name"])
            model_version = model_info["version"]
            model_name_full = f"{model_name}-v{model_version}"
            model_state = model_info["state"]

            if model_state == "READY":
                self.add_label(model_name_full)
            else:
                self.remove_label(model_name_full)
                
        self.logger.info("Completed label sync for all models",
                        pod=self.pod_name)

    def add_label(self, model_name: str):
        """
        Add a label to the Triton server indicating a model is loaded
        Format: sonic.model.loaded/modelname-v1: true
        """
        label_key = format_model_label(model_name)
        self.logger.info("Adding model label to server", 
                        label=label_key,
                        pod=self.pod_name)
        
        try:
            # Check if label already exists
            if self.pod.metadata.labels and label_key in self.pod.metadata.labels:
                self.logger.warning("Model label already exists", 
                                  model=model_name,
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
        self.logger.info("Removing model label from server", 
                        label=label_key,
                        pod=self.pod_name)
        
        try:
            # Get fresh pod data to ensure we have current labels
            current_pod = self.v1.read_namespaced_pod(
                name=self.pod_name,
                namespace=self.pod_namespace
            )
            
            if current_pod.metadata.labels and label_key in current_pod.metadata.labels:
                # Create a new labels dict and set the label value to None
                new_labels = dict(current_pod.metadata.labels)
                new_labels[label_key] = None
                
                # Update the pod with new labels
                self.v1.patch_namespaced_pod(
                    name=self.pod_name,
                    namespace=self.pod_namespace,
                    body={"metadata": {"labels": new_labels}}
                )

            else:
                self.logger.warning("Model label not found", 
                                  model=model_name,
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
            local_port = self.setup_port_forward(8002)
            
            base_url = f"http://localhost:{local_port}"
            
            response = self.query_triton_server(f"{base_url}/metrics", method="GET")
            
            gpu_memory = {}
            
            for line in response.text.split('\n'):
                # Skip comments and empty lines
                if line.startswith('#') or not line.strip():
                    continue
                
                # Parse metric line
                try:
                    # Format: metric_name{label1="value1",label2="value2"} value
                    metric_name, value = line.split(' ', 1)
                    value = float(value)
                    
                    labels = {}
                    if '{' in metric_name:
                        name, label_str = metric_name.split('{', 1)
                        label_str = label_str.rstrip('}')
                        for label in label_str.split(','):
                            key, val = label.split('=', 1)
                            labels[key] = val.strip('"')
                        metric_name = name
                    else:
                        name = metric_name
                    
                    # Process GPU memory metrics
                    if metric_name == 'nv_gpu_memory_total_bytes':
                        gpu_uuid = labels.get('gpu_uuid')
                        if gpu_uuid is not None:
                            if gpu_uuid not in gpu_memory:
                                gpu_memory[gpu_uuid] = {}
                            gpu_memory[gpu_uuid]['total_memory'] = int(value)
                    
                    elif metric_name == 'nv_gpu_memory_used_bytes':
                        gpu_uuid = labels.get('gpu_uuid')
                        if gpu_uuid is not None:
                            if gpu_uuid not in gpu_memory:
                                gpu_memory[gpu_uuid] = {}
                            gpu_memory[gpu_uuid]['used_memory'] = int(value)
                    
                    # Log other interesting metrics
                    elif metric_name in ['nv_gpu_utilization', 'nv_gpu_power_usage', 'nv_gpu_temperature']:
                        gpu_uuid = labels.get('gpu_uuid')
                        if gpu_uuid is not None:
                            self.logger.debug(f"GPU {metric_name}", 
                                           gpu_uuid=gpu_uuid,
                                           value=value,
                                           pod=self.pod_name)
                
                except Exception as e:
                    self.logger.warning("Failed to parse metric line", 
                                      line=line,
                                      error=str(e))
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
            
        except requests.exceptions.RequestException as e:
            self.logger.error("Failed to query Triton metrics API", 
                            error=str(e),
                            pod=self.pod_name)
            raise
        except Exception as e:
            self.logger.error("Unexpected error while getting GPU memory", 
                            error=str(e),
                            pod=self.pod_name)
            raise
        finally:
            self.cleanup_port_forward()