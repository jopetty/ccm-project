# ccm-project

## Installation Steps

1. Clone this repo to `/scratch/NETID` and `cd` into it.
2. Move to an interactive job node:
```bash
srun --pty /bin/bash
```

3. Copy the following singularity overlay:
```bash
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz .
```

4. Extract the gzipped overlay:
```bash
gunzip overlay-15GB-500K.ext3.gz
```

5. Launch the container in read/write mode:
```bash
singularity exec --overlay overlay-15GB-500K.ext3:rw /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash
```

6. Install Miniconda
```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3 && rm Miniconda3-latest-Linux-x86_64.sh
```

7. Create the following script at `/ext3/env.sh`:
```bash
#!/bin/bash

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PYTHONPATH=/ext3/miniconda3/bin:$PATH
```

8. Activate the conda base environment
```bash
source /ext3/env.sh
```
9. Install packages we need.
```bash
conda env create
```
10. Exit singularity with `exit`