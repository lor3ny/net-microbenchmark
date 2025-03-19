# Modules load and any other needed command
export UCX_LOG_LEVEL=error

# Optional variables specifying binary paths for applications

export SLURM_CPU_BIND=socket
export UCX_NET_DEVICES=mlx5_0:1

if [ -n "${UCX_IB_SL}" ]; then 
    export UCX_IB_SL=${UCX_IB_SL}
else
    export UCX_IB_SL=1
fi
export UCX_PROTO_ENABLE=y
