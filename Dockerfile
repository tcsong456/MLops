FROM conda/miniconda3

LABEL org.label-schema.vendor = "Microsoft" \
    org.label-schema.url = "https://hub.docker.com/r/microsoft/mlopspython" \
    org.label-schema.vcs-url = "https://github.com/microsoft/MLOpsPython"

#COPY diabetes_regression/ci_dependencies.yml /setup/
COPY diabetes/ci_dependencies.yml /env_steup/

# activate environment,add /usr/local/envs/mlopspython_ci/bin 
#to environment variables for current session
ENV PATH /usr/local/envs/mlopspython_ci/bin:$PATH

RUN conda update -n base -c defaults conda && \
    conda install python=3.7.5 && \
    conda env create -f /env_steup/ci_dependencies.yml && \
    /bin/bash -c "source activate mlopspython_ci" && \
    az --version && \
    chmod -R 777 /usr/local/envs/mlopspython_ci/lib/python3.7
