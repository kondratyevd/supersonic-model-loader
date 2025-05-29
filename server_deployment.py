from kubernetes import client, config
from kubernetes.client.rest import ApiException
from logger import get_logger
from typing import List, Dict
from server import Server
import tritonclient.grpc as grpcclient
from tritonclient.grpc.service_pb2 import RepositoryIndexResponse

class ServerDeployment:
    def __init__(self, release_name: str, namespace: str):
        self.release_name = release_name
        self.namespace = namespace
        self.logger = get_logger("server")
        self.v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        
    def get_deployment(self):
        """
        Get the deployment by name in the specified namespace
        """
        deployment_name = f"{self.release_name}-triton"
        self.logger.info("Fetching deployment", 
                        name=deployment_name,
                        namespace=self.namespace)
        
        try:
            deployment = self.v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            self.logger.info("Deployment retrieved", 
                           name=deployment_name,
                           replicas_available=deployment.status.available_replicas,
                           replicas_ready=deployment.status.ready_replicas,
                           replicas_total=deployment.spec.replicas)
            
            return deployment
            
        except ApiException as e:
            if e.status == 404:
                self.logger.error("Deployment not found", 
                                name=deployment_name,
                                namespace=self.namespace)
                raise Exception(f"Deployment {deployment_name} not found in namespace {self.namespace}")
            
            self.logger.error("Failed to get deployment", 
                            name=deployment_name,
                            namespace=self.namespace,
                            error=str(e))
            raise e

    def get_servers(self) -> List['Server']:
        """
        Get all Triton server pods in the deployment
        """
        deployment_name = f"{self.release_name}-triton"
        self.logger.info("Fetching server pods", 
                        deployment=deployment_name,
                        namespace=self.namespace)
        
        try:
            # Get all pods for this deployment
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app.kubernetes.io/instance={self.release_name},app.kubernetes.io/name=supersonic,app.kubernetes.io/component=triton"
            ).items
            
            servers = []
            for pod in pods:
                server = Server(pod)
                servers.append(server)
                
            self.logger.info("Found server pods", 
                           count=len(servers),
                           deployment=deployment_name)
            
            return servers
            
        except Exception as e:
            self.logger.error("Failed to get server pods", 
                            deployment=deployment_name,
                            error=str(e))
            raise

    def scale(self, replicas: int):
        """
        Scale the number of Triton servers in the deployment
        """
        deployment_name = f"{self.release_name}-triton"
        self.logger.info("Scaling deployment", 
                        deployment=deployment_name,
                        current_replicas=len(self.get_servers()),
                        target_replicas=replicas)
        
        try:
            # Create a patch to update the replicas
            patch = {
                "spec": {
                    "replicas": replicas
                }
            }
            
            # Apply the patch to scale the deployment
            self.v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=patch
            )
            
            self.logger.info("Successfully scaled deployment",
                           deployment=deployment_name,
                           target_replicas=replicas)
            
        except ApiException as e:
            self.logger.error("Failed to scale deployment",
                            deployment=deployment_name,
                            target_replicas=replicas,
                            error=str(e))
            raise

    def get_aggregated_model_repository_index(self) -> RepositoryIndexResponse:
        """
        Get and aggregate model repository indices from all Triton servers in the deployment.
        Returns a merged RepositoryIndexResponse containing unique models from all servers.
        """
        self.logger.info("Aggregating model repository indices from all servers")
        
        # Get all servers
        servers = self.get_servers()
        
        # Create merged response
        merged = RepositoryIndexResponse()
        
        # Dictionary to track models by name
        # Each model name maps to a dict of version -> model
        models_by_name = {}
        
        for server in servers:
            try:
                # Get models from this server
                local_port = server.setup_port_forward(8001)  # gRPC port
                client = server.get_triton_client(local_port)
                
                # Get repository index
                repository_index = client.get_model_repository_index()
                
                # Process each model
                for model in repository_index.models:
                    model_name = model.name
                    
                    # Initialize dict for this model name if not exists
                    if model_name not in models_by_name:
                        models_by_name[model_name] = {}
                    
                    # If model has a version, add/update it in the versions dict
                    if model.version:
                        models_by_name[model_name][model.version] = model
                    # If model has no version, only add it if we don't have any versions yet
                    elif not models_by_name[model_name]:
                        models_by_name[model_name][""] = model
                
            except Exception as e:
                self.logger.error("Failed to get model repository index from server",
                                error=str(e),
                                pod=server.pod_name)
                continue
            finally:
                server.cleanup_port_forward()
        
        # Clean up unversioned entries if versions exist
        for model_versions in models_by_name.values():
            if len(model_versions) > 1 and "" in model_versions:
                del model_versions[""]
        
        # Add all models to the merged response
        for model_versions in models_by_name.values():
            merged.models.extend(model_versions.values())
        
        # print(merged)
        self.logger.info("Successfully aggregated model repository indices",
                        total_models=len(merged.models))
        
        return merged