#!/usr/bin/env python3

# import yaml
import argparse
import os
import stat
import subprocess
from pathlib import Path
from time import sleep


class SlurmJob:
    def __init__(
        self,
        name,
        time,
        gpu,
        partition,
        num_gpus,
        num_cpus,
        memory,
        email,
        unk_args,
        interactive,
        cuda_version,
        exclude_nodes,
        config_file="",
        index=0,
    ):
        host_name = os.environ.get("HOSTNAME", "")
        self.on_tue_cluster = "bg-slurmb" in host_name

        self.name = f"{name}-{index}"
        self.email = email
        self.time = time

        days, hours, minutes = list(map(int, [time.split("-")[0]] + time.split("-")[1].split(":")))
        self.gpu = f"A100:{num_gpus}" if gpu == "A100" else num_gpus
        self.partition = partition
        self.num_cpus = num_cpus
        self.memory = memory
        self.run_args = " ".join(unk_args)
        self.config_file = config_file
        # self.cuda_version = "11.7" if gpu == "A100" else "11.2"
        self.cuda_version = cuda_version
        self.interactive = interactive
        self.exclude_nodes = exclude_nodes
        self.src_dir = "$HOME/projects"
        self.local_dir = "/local/$USER"
        self.scratch_emmy_dir = "/scratch-emmy/usr/$USER" if gpu == "A100" else "/scratch/usr/$USER"
        self.scratch_dir = "/scratch/usr/$USER"
        self.project_dir_name = "nnsysident"
        self.singularity_run_command = self.get_singularity_run_command()

    @property
    def resource_config_string(self):
        if not Path("logs").exists():
            os.mkdir("logs")
        config_string = f"""
#SBATCH --job-name={self.name}                   # Name of the job
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --cpus-per-task={self.num_cpus}          # Number of CPU cores per task
#SBATCH --nodes=1                                # Ensure that all cores are on one machine
#SBATCH --time={self.time}                       # Runtime in D-HH:MM
#SBATCH --mem-per-cpu={self.memory}              # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/{self.name}.%j.out         # File to which STDOUT will be written
#SBATCH --error=logs/{self.name}.%j.err          # File to which STDERR will be written
#SBATCH --mail-type=ALL                          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user={self.email}                 # Email to which notifications will be sent
#SBATCH -p {self.partition}                      # Partition to submit to
#SBATCH -G {self.gpu}                            # Number of requested GPUs
#SBATCH --exclude={self.exclude_nodes}           # Exclude nodes
#SBATCH --constraint=inet                        # Access to internet
            """
        return config_string

    def get_singularity_run_command(self):
        run_cmd = f"""
            export SRCDIR={self.src_dir}
            export JOBDIR={self.local_dir}
            export JOBOUTDIR={self.scratch_emmy_dir}/outputs/$SLURM_JOB_ID
            mkdir -p $JOBDIR
            mkdir -p $JOBOUTDIR
        """

        run_cmd += f"""
        module load singularity
        module load cuda/{self.cuda_version}
        scontrol show job $SLURM_JOB_ID  # print some info
        """

        if self.config_file:
            run_cmd += f"""
                cp {self.config_file} $JOBDIR/config.yaml
            """
            self.run_args += " --experiment-file $HOME/config.yaml"

        singularity_mode = "instance start" if self.interactive else "exec"
        instance_name = self.name if self.interactive else ""

        bindings = {self.scratch_dir: "/project/data/", f"$HOME/projects/{self.project_dir_name}": "/project/"}
        bindings = ",".join([f"{k}:{v}" for k, v in bindings.items()])
        run_cmd += f""" 
            singularity {singularity_mode} \
            --nv \
            --env-file .env \
            --env inside_singularity_container=YES \
            --no-home  \
            --bind {bindings}  \
            singularity.sif {instance_name} \
            /project/run.py
            """
        if self.interactive:
            run_cmd += "sleep infinity"
        return run_cmd

    def run(self):
        slurm_job_bash_file = f"./{self.name}.sh"
        slurm_job_bash_file_content = (
            "#!/bin/bash \n \n" + self.resource_config_string + "\n" + self.singularity_run_command
        )
        with open(slurm_job_bash_file, "w") as f:
            f.write(slurm_job_bash_file_content)

        os.chmod(slurm_job_bash_file, stat.S_IRWXU)

        try:
            output = subprocess.check_output("sbatch " + slurm_job_bash_file, shell=True)
            sleep(5)
            job_id = int(output[20:].strip())
            node = subprocess.check_output(f"scontrol show job {job_id}| grep ' NodeList'", shell=True).strip()
            node = str(node).split("=")[1][:-1]
            print(f"Successfully submitted job with ID {job_id} to node {node}.")
            # print(self.resource_config_string)
            # print(self.singularity_run_command)
        finally:
            # remove the bash file
            os.remove(slurm_job_bash_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running jobs on SLURM cluster")
    parser.add_argument(
        "--name",
        dest="name",
        action="store",
        default="noname",
        type=str,
        help="",
    )
    parser.add_argument(
        "--njobs",
        dest="num_jobs",
        action="store",
        default=1,
        type=int,
        help="",
    )
    parser.add_argument(
        "--time",
        dest="time",
        action="store",
        default="0-01:00",
        type=str,
        help="time to complete each job. Specify in the following format: D-HH:MM",
    )
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store",
        default="A100",
        type=str,
        help="",
    )
    parser.add_argument(
        "--partition",
        dest="partition",
        action="store",
        default="grete:shared",
        type=str,
        help="",
    )
    parser.add_argument(
        "--ncpus",
        dest="num_cpus",
        action="store",
        default=4,
        type=int,
        help="",
    )
    parser.add_argument(
        "--ngpus",
        dest="num_gpus",
        action="store",
        default=1,
        type=int,
        help="",
    )
    parser.add_argument(
        "--memory",
        dest="memory",
        action="store",
        default="3G",
        type=str,
        help="",
    )
    parser.add_argument(
        "--email",
        dest="email",
        action="store",
        default=os.getenv("EMAIL"),
        type=str,
        help="",
    )
    parser.add_argument(
        "--cuda",
        dest="cuda_version",
        action="store",
        default="11.7",
        type=str,
        help="Cuda version to use",
    )
    parser.add_argument(
        "--exclude",
        dest="exclude_nodes",
        action="store",
        default="",
        type=str,
        help="List of nodes to exclude",
    )
    parser.add_argument("--interactive", dest="interactive", action="store_true", default=False, help="")
    args, unk_args = parser.parse_known_args()

    for job_index in range(args.num_jobs):
        job = SlurmJob(
            name=args.name,
            time=args.time,
            gpu=args.gpu,
            partition=args.partition,
            num_gpus=args.num_gpus,
            num_cpus=args.num_cpus,
            memory=args.memory,
            email=args.email,
            unk_args=unk_args,
            interactive=args.interactive,
            cuda_version=args.cuda_version,
            exclude_nodes=args.exclude_nodes,
            index=job_index,
        )
        job.run()
