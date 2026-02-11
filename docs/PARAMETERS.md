# Parameter Mapping (Paper ↔ Config)

Update `configs/paper.yml` whenever you change scenario parameters in the paper.

Suggested mapping (Table III ↔ config):
- latency_mean_s, latency_std_s, packet_loss, retries
- queue_timeout_s, e2e_timeout_s
- bank capacity, service_time_s, local_time_s
- degradation settings (S1 only)
