from kubernetes import client, config
from kubernetes.client.rest import ApiException
from logger import get_logger
from typing import List
from server import Server

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
        Returns:
            V1Deployment: The deployment object if found
        Raises:
            ApiException: If the deployment is not found or other API errors occur
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
        Returns:
            List[Server]: List of Server objects representing each pod
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
        Args:
            replicas (int): The desired number of replicas
        Raises:
            ApiException: If scaling fails
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