# SuperSONIC: dynamic model loading

This repo includes developments for dynamic model loading in [SuperSONIC](https://github.com/fastmachinelearning/SuperSONIC) inference platform.

### Intended functionality:
- All ML models are NOT necessarily loaded into all Triton servers.
- Instead of using a single load balancer over all Triton servers, inference requests will be routed via model-specific load balancers across only those Triton servers where a given model is loaded.
- In main SuperSONIC repo, Envoy Proxy will be configured to extract model name from gRPC request body and use it to reroute the request to the load balancer corresponding to that model.
- Models currently present on a Triton server will be represented via labels on the Kubernetes pod that hosts the server. This will allow load balancers to automatically adjust address pools when models are loaded and unloaded from servers.
- Decision logic for loading and unloading models will be based on GPU memory and load. It will be implemented somewhere in this repo.

### Caveats

- Kubernetes resources can only contain alphanumerical characters and hyphens in their names. We need to be careful when creating resources and labels based on model names.
