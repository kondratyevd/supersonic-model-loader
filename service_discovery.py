from utils import format_model_label
from logger import get_logger
from kubernetes import client

class Service:
    def __init__(self, model_name_full: str, release_name: str, namespace: str):
        self.model_name_full = model_name_full
        self.label_key = format_model_label(model_name_full)
        self.release_name = release_name
        self.namespace = namespace
        self.service_name = f"{self.release_name}-{self.model_name_full}"
        self.logger = get_logger(f"service")
        self.endpoints = set()
        self.v1 = client.CoreV1Api()
        
        self.logger.debug("Service created", 
                         model=self.model_name_full)
    
    def spawn(self):
        """
        Spawn a headless Kubernetes Service
        """
        self.logger.info("Spawning Service", 
                        model=self.model_name_full)
        
        try:
            # Check if service already exists
            try:
                existing_service = self.v1.read_namespaced_service(
                    name=self.service_name,
                    namespace=self.namespace
                )
                self.logger.warning("Service already exists", 
                                  name=self.service_name,
                                  namespace=self.namespace)
                return
            except client.exceptions.ApiException as e:
                if e.status != 404:  # If error is not "not found", re-raise
                    raise
            
            # Create service metadata
            metadata = client.V1ObjectMeta(
                name=self.service_name,
                labels={
                    "app.kubernetes.io/name": "supersonic",
                    "app.kubernetes.io/instance": self.release_name,
                    "app.kubernetes.io/component": "triton",
                }
            )
            
            # Create service ports
            ports = [
                client.V1ServicePort(
                    name="http",
                    port=8000,
                    target_port=8000,
                    protocol="TCP"
                ),
                client.V1ServicePort(
                    name="grpc",
                    port=8001,
                    target_port=8001,
                    protocol="TCP"
                ),
                client.V1ServicePort(
                    name="metrics",
                    port=8002,
                    target_port=8002,
                    protocol="TCP"
                )
            ]
            
            # Create service spec
            spec = client.V1ServiceSpec(
                cluster_ip="None",  # Headless service
                ports=ports,
                selector={
                    "app.kubernetes.io/name": "supersonic",
                    "app.kubernetes.io/instance": self.release_name,
                    "app.kubernetes.io/component": "triton",
                    self.label_key: "true"
                }
            )
            
            # Create the service
            service = client.V1Service(
                api_version="v1",
                kind="Service",
                metadata=metadata,
                spec=spec
            )
            
            # Create the service in Kubernetes
            self.v1.create_namespaced_service(
                namespace=self.namespace,
                body=service
            )
            
            self.logger.info("Service created successfully",
                           name=self.service_name,
                           ports=[p.name for p in ports])
            
        except Exception as e:
            self.logger.error("Failed to create Service",
                            model=self.service_name,
                            error=str(e))
            raise