from utils import format_model_label, escape_model_name
from logger import get_logger
from kubernetes import client, config
import requests
from typing import Dict, List, Optional
import threading
import queue
import time
import subprocess
import os
import signal
import atexit
import json

class Server:
    def __init__(self, pod):
        self.logger = get_logger("server")
        self.pod = pod
        self.v1 = client.CoreV1Api()
        self.port_forward_process = None
        atexit.register(self.cleanup_port_forward)
        self.triton_models_info = {}
        # Get release name from pod labels
        self.release_name = self.pod.metadata.labels.get("app.kubernetes.io/instance", "supersonic-test")

    def cleanup_port_forward(self):
        """Clean up port-forward process if it exists"""
        if self.port_forward_process:
            try:
                os.killpg(os.getpgid(self.port_forward_process.pid), signal.SIGTERM)
            except:
                pass

    def get_models(self) -> Dict[str, Dict]:
        """
        Get the models loaded on the Triton Inference Server API.
        Uses kubectl port-forward for direct pod access.
        
        Returns:
            Dict[str, Dict]: Dictionary containing model information with the following structure:
            {
                "model_name": {
                    "name": str,
                    "version": str,
                    "state": str,
                    "status": Dict
                }
            }
        """
        self.logger.info("Querying Triton server for loaded models", 
                        pod=self.pod.metadata.name)
        
        try:
            # Create a queue to receive the port number
            port_queue = queue.Queue()
            
            # Start port-forwarding in a separate thread
            def port_forward():
                try:
                    # Use kubectl port-forward
                    cmd = [
                        "kubectl", "port-forward",
                        f"pod/{self.pod.metadata.name}",
                        "8000:8000",
                        "-n", self.pod.metadata.namespace
                    ]
                    
                    # Start the process in a new process group
                    self.port_forward_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        preexec_fn=os.setsid
                    )
                    
                    # Wait a moment for the port-forward to establish
                    time.sleep(2)
                    
                    # Check if process is still running
                    if self.port_forward_process.poll() is not None:
                        stderr = self.port_forward_process.stderr.read().decode()
                        raise Exception(f"Port-forward failed: {stderr}")
                    
                    # Use localhost:8000
                    port_queue.put(8000)
                    
                    # Keep the connection open
                    while True:
                        if self.port_forward_process.poll() is not None:
                            stderr = self.port_forward_process.stderr.read().decode()
                            raise Exception(f"Port-forward terminated: {stderr}")
                        time.sleep(0.1)
                except Exception as e:
                    self.logger.warning("Port-forwarding terminated", 
                                    pod=self.pod.metadata.name)
                    port_queue.put(None)
            
            # Start port-forwarding thread
            pf_thread = threading.Thread(target=port_forward, daemon=True)
            pf_thread.start()
            
            # Wait for the port number
            local_port = port_queue.get()
            if local_port is None:
                raise Exception("Failed to establish port-forwarding")
            
            self.logger.info("Port-forwarding established", 
                           pod=self.pod.metadata.name,
                           local_port=local_port)
            
            # Use localhost with the forwarded port
            base_url = f"http://localhost:{local_port}"
            
            # Set up common headers for Triton API
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # Query Triton model repository status
            response = requests.post(
                f"{base_url}/v2/repository/index",
                headers=headers
            )
            response.raise_for_status()
            
            # Log raw response for debugging
            self.logger.debug("Raw repository index response", 
                            status_code=response.status_code,
                            content=response.text)
            
            try:
                # The response is a list of model information dictionaries
                models_data = response.json()
            except json.JSONDecodeError as e:
                self.logger.error("Invalid JSON response from repository index", 
                                error=str(e),
                                content=response.text)
                raise
            
            self.logger.debug("Received model data from repository", 
                            models_data=models_data)
            
            models_info = {}
            
            # Process each model in the repository
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
                
                # Get detailed model status
                status_response = requests.get(
                    f"{base_url}/v2/models/{model_name}/ready",
                    headers=headers
                )
                status_response.raise_for_status()
                
                # The ready endpoint returns a boolean value directly
                is_ready = status_response.text.lower() == 'true'
                
                # Get model config
                config_response = requests.get(
                    f"{base_url}/v2/models/{model_name}/config",
                    headers=headers
                )
                config_response.raise_for_status()
                
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
                    # "status": {"ready": is_ready},
                    # "config": config_data
                }
                
                self.logger.info("Model information retrieved", 
                               model=model_name,
                               version=self.triton_models_info[model_name]["version"],
                               state=self.triton_models_info[model_name]["state"],
                               pod=self.pod.metadata.name)
            
            self.logger.info("Successfully retrieved model information", 
                           count=len(self.triton_models_info),
                           pod=self.pod.metadata.name)
            
        except requests.exceptions.RequestException as e:
            self.logger.error("Failed to query Triton server API", 
                            error=str(e),
                            pod=self.pod.metadata.name)
            raise
        except Exception as e:
            self.logger.error("Unexpected error while getting models", 
                            error=str(e),
                            pod=self.pod.metadata.name)
            raise
        finally:
            # Clean up port-forward process
            self.cleanup_port_forward()
    
    def sync_labels(self):
        """
        Sync the labels on the Triton server with the labels on the local pod
        """
        self.logger.info("Starting label sync for models", 
                        pod=self.pod.metadata.name,
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
                        pod=self.pod.metadata.name)

    def add_label(self, model_name: str):
        """
        Add a label to the Triton server indicating a model is loaded
        Format: sonic.model.loaded/modelname-v1: true
        """
        label_key = format_model_label(model_name)
        self.logger.info("Adding model label to server", 
                        # model=model_name,
                        label=label_key,
                        pod=self.pod.metadata.name)
        
        try:
            # Check if label already exists
            if self.pod.metadata.labels and label_key in self.pod.metadata.labels:
                self.logger.warning("Model label already exists", 
                                  model=model_name,
                                  label=label_key,
                                  pod=self.pod.metadata.name)
                return
            
            # Add the new label
            if self.pod.metadata.labels is None:
                self.pod.metadata.labels = {}
            
            self.pod.metadata.labels[label_key] = "true"
            
            # Update the pod
            self.v1.patch_namespaced_pod(
                name=self.pod.metadata.name,
                namespace=self.pod.metadata.namespace,
                body={"metadata": {"labels": self.pod.metadata.labels}}
            )
            
        except Exception as e:
            self.logger.error("Failed to add model label", 
                            model=model_name,
                            error=str(e),
                            pod=self.pod.metadata.name)
            raise

    def remove_label(self, model_name: str):
        """
        Remove a label from the Triton server
        """
        label_key = format_model_label(model_name)
        self.logger.info("Removing model label from server", 
                        # model=model_name,
                        label=label_key,
                        pod=self.pod.metadata.name)
        
        try:
            # Get fresh pod data to ensure we have current labels
            current_pod = self.v1.read_namespaced_pod(
                name=self.pod.metadata.name,
                namespace=self.pod.metadata.namespace
            )
            
            if current_pod.metadata.labels and label_key in current_pod.metadata.labels:
                # Create a new labels dict and set the label value to None
                new_labels = dict(current_pod.metadata.labels)
                new_labels[label_key] = None
                
                # Update the pod with new labels
                self.v1.patch_namespaced_pod(
                    name=self.pod.metadata.name,
                    namespace=self.pod.metadata.namespace,
                    body={"metadata": {"labels": new_labels}}
                )

            else:
                self.logger.warning("Model label not found", 
                                  model=model_name,
                                  label=label_key,
                                  pod=self.pod.metadata.name)
                
        except Exception as e:
            self.logger.error("Failed to remove model label", 
                            model=model_name,
                            error=str(e),
                            pod=self.pod.metadata.name)
            raise

    def find_label(self, model_name: str) -> bool:
        """
        Check if Triton server has a model label
        Returns:
            bool: True if the label exists, False otherwise
        """
        label_key = format_model_label(model_name)
        self.logger.debug("Checking for model label", 
                         model=model_name,
                         label=label_key,
                         pod=self.pod.metadata.name)
        
        try:
            has_label = (self.pod.metadata.labels is not None and 
                        label_key in self.pod.metadata.labels and 
                        self.pod.metadata.labels[label_key] == "true")
            
            self.logger.debug("Model label check completed", 
                            model=model_name,
                            label=label_key,
                            found=has_label,
                            pod=self.pod.metadata.name)
            
            return has_label
            
        except Exception as e:
            self.logger.error("Failed to check model label", 
                            model=model_name,
                            error=str(e),
                            pod=self.pod.metadata.name)
            raise

    def get_gpu_memory(self) -> Dict[str, Dict[str, int]]:
        """
        Get GPU memory information from the Triton server.
        
        Returns:
            Dict[str, Dict[str, int]]: Dictionary containing GPU memory information:
            {
                "gpu_uuid": {
                    "total_memory": int,  # Total memory in bytes
                    "free_memory": int,   # Free memory in bytes
                    "used_memory": int    # Used memory in bytes
                }
            }
        """
        self.logger.info("Querying GPU memory information", 
                        pod=self.pod.metadata.name)
        
        try:
            # Create a queue to receive the port number
            port_queue = queue.Queue()
            
            # Start port-forwarding in a separate thread
            def port_forward():
                try:
                    # Use kubectl port-forward
                    cmd = [
                        "kubectl", "port-forward",
                        f"pod/{self.pod.metadata.name}",
                        "8002:8002",
                        "-n", self.pod.metadata.namespace
                    ]
                    
                    # Start the process in a new process group
                    self.port_forward_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        preexec_fn=os.setsid
                    )
                    
                    # Wait a moment for the port-forward to establish
                    time.sleep(2)
                    
                    # Check if process is still running
                    if self.port_forward_process.poll() is not None:
                        stderr = self.port_forward_process.stderr.read().decode()
                        raise Exception(f"Port-forward failed: {stderr}")
                    
                    # Use localhost:8002
                    port_queue.put(8002)
                    
                    # Keep the connection open
                    while True:
                        if self.port_forward_process.poll() is not None:
                            stderr = self.port_forward_process.stderr.read().decode()
                            raise Exception(f"Port-forward terminated: {stderr}")
                        time.sleep(0.1)
                except Exception as e:
                    self.logger.warning("Port-forwarding terminated", 
                                      pod=self.pod.metadata.name)
                    port_queue.put(None)
            
            # Start port-forwarding thread
            pf_thread = threading.Thread(target=port_forward, daemon=True)
            pf_thread.start()
            
            # Wait for the port number
            local_port = port_queue.get()
            if local_port is None:
                raise Exception("Failed to establish port-forwarding")
            
            self.logger.info("Port-forwarding established", 
                           pod=self.pod.metadata.name,
                           local_port=local_port)
            
            # Use localhost with the forwarded port
            base_url = f"http://localhost:{local_port}"
            
            # Query Triton metrics endpoint
            response = requests.get(
                f"{base_url}/metrics",
                headers={'Accept': 'text/plain'}
            )
            response.raise_for_status()
            
            gpu_memory = {}
            
            # Parse Prometheus format metrics
            for line in response.text.split('\n'):
                # Skip comments and empty lines
                if line.startswith('#') or not line.strip():
                    continue
                
                # Parse metric line
                try:
                    # Format: metric_name{label1="value1",label2="value2"} value
                    metric_name, value = line.split(' ', 1)
                    value = float(value)
                    
                    # Extract labels
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
                                           pod=self.pod.metadata.name)
                
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
                                   pod=self.pod.metadata.name)
            
            return gpu_memory
            
        except requests.exceptions.RequestException as e:
            self.logger.error("Failed to query Triton metrics API", 
                            error=str(e),
                            pod=self.pod.metadata.name)
            raise
        except Exception as e:
            self.logger.error("Unexpected error while getting GPU memory", 
                            error=str(e),
                            pod=self.pod.metadata.name)
            raise
        finally:
            # Clean up port-forward process
            self.cleanup_port_forward()