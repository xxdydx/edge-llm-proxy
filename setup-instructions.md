We can give you an on-demand GPU development machine through our Lumid FlowMesh
cluster. I would start with a single RTX 5080 (16 GB VRAM), 8 CPU cores, 32 GB
RAM, and about 80 GB of local scratch disk. That should be a good fit for the
edge-client work.

1. Create a Lumid access token. Go to https://lum.id, choose New token, then
   Custom (per service), and enable Write for flowmesh..
2. Install the FlowMesh CLI with one of the following commands. You will need
   Python 3.12 or newer:. pip install "flowmesh[cli]" pipx install
   "flowmesh[cli]" uv tool install "flowmesh[cli]"
3. Configure the CLI.

flowmesh init https://lum.id/fm --api-key <flowmesh-api-key> flowmesh config 4.
Create a file named ssh-workflow.yaml with the following content and get your
SSH public key. Paste the public key into the YAML below in place of
<your-public-ssh-key>:

apiVersion: flowmesh/v1 kind: SSHTask metadata: name: arul-edge-client-dev
annotations: description: Development environment for the edge LLM client
project

spec: taskType: ssh interactive: true user: flowmesh authorizedKeys: -
<your-public-ssh-key> resources: hardware: gpu: type: RTX 5080 count: 1 cpu: 8
memory: 32Gi ttlSeconds: 28800 accessMode: forward output: destination: type:
http 5. Submit it:

flowmesh workflow submit ssh-workflow.yaml The output will include the task ID.
Connect with:

flowmesh ssh connect <task-id> After connecting, run these once to confirm the
machine:

nvidia-smi nproc free -h df -h If you disconnect, the session is still there
until its TTL expires. Reconnect with the same flowmesh ssh connect <task-id>
command. When you are done, release it with:

flowmesh task stop <task-id> One important limitation: these are disposable SSH
containers. The hard maximum session TTL is eight hours, although the cluster
can impose a shorter cap. When it ends, the container and its local filesystem
are removed. Please keep code in Git and push anything important before then.
Each new session starts clean, so you will need to clone the repo and set up
your environment again. Once you have a setup that you like, we can avoid most
of the repeated startup time by putting it in a Docker image and adding its
reference to the task:

image: <your-registry-image:tag> The image needs to be in a registry the worker
can pull from, and it should be built from the FlowMesh GPU SSH-session base
image so it retains the SSH server and entrypoint.

Let me know if you have any issues during the setup.

My Lumid Token: LUMID_TOKEN at .env
