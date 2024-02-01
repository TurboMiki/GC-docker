FROM python:3.8-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools && python -m pip install --upgrade pip



COPY --chown=user:user requirements.txt /opt/app/
RUN python -m piptools sync requirements.txt



COPY --chown=user:user custom_algorithm.py /opt/app/
COPY --chown=user:user process.py /opt/app/

# This is the checkpoint file, uncomment the line below and modify /local/path/to/the/checkpoint to your needs
COPY --chown=algorithm:algorithm 192_224_fz_sb16_N_latest.pth /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
