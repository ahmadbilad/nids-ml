"""PCAP reader and flow feature extractor for network traffic replay."""

from collections import defaultdict
from pathlib import Path
import pandas as pd
from scapy.all import IP, TCP, UDP, rdpcap

from app.model_utils import FEATURE_COLUMNS


def extract_flows_from_pcap(pcap_path, max_packets=5000):
    """
    Parse a PCAP file, aggregate packet streams into bidirectional flows,
    and compute standard flow features matching the NSL-KDD schema.
    """
    packets = rdpcap(str(pcap_path), count=max_packets)
    if not packets:
        raise ValueError("PCAP file contains no readable packets.")

    flows = defaultdict(list)

    # 1. Group into bidirectional flows
    for pkt in packets:
        if IP not in pkt:
            continue

        ip_src = pkt[IP].src
        ip_dst = pkt[IP].dst
        proto_num = pkt[IP].proto
        proto_str = "tcp" if proto_num == 6 else ("udp" if proto_num == 17 else "icmp")

        sport = 0
        dport = 0
        flags = ""

        if TCP in pkt:
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
            flags = str(pkt[TCP].flags)
        elif UDP in pkt:
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport

        # Bidirectional flow key (canonical sorted direction)
        if (ip_src, sport) < (ip_dst, dport):
            flow_key = (ip_src, ip_dst, sport, dport, proto_str)
            direction = "forward"
        else:
            flow_key = (ip_dst, ip_src, dport, sport, proto_str)
            direction = "backward"

        pkt_len = len(pkt)
        flows[flow_key].append({
            "time": float(pkt.time),
            "len": pkt_len,
            "direction": direction,
            "flags": flags,
            "dport": dport,
            "sport": sport,
            "proto": proto_str,
        })

    # 2. Compute per-flow statistics
    flow_records = []
    for (src, dst, sp, dp, proto), pkts in flows.items():
        start_time = pkts[0]["time"]
        end_time = pkts[-1]["time"]
        duration = max(0.0, end_time - start_time)

        fwd_pkts = [p for p in pkts if p["direction"] == "forward"]
        bwd_pkts = [p for p in pkts if p["direction"] == "backward"]

        src_bytes = sum(p["len"] for p in fwd_pkts)
        dst_bytes = sum(p["len"] for p in bwd_pkts)

        # Service detection heuristic based on destination port
        service_map = {80: "http", 443: "http", 21: "ftp", 22: "ssh", 25: "smtp", 53: "domain_u"}
        service = service_map.get(dp, service_map.get(sp, "private"))

        # TCP Flag abstraction
        all_flags = "".join(p["flags"] for p in pkts)
        if "S" in all_flags and "A" not in all_flags:
            flag = "S0"
        elif "R" in all_flags:
            flag = "REJ"
        else:
            flag = "SF"

        flow_count = len(pkts)
        logged_in = 1 if (service in ["http", "smtp"] and "A" in all_flags) else 0

        # Construct row matching 41 FEATURE_COLUMNS
        row = {col: 0.0 for col in FEATURE_COLUMNS}
        row.update({
            "duration": round(duration, 2),
            "protocol_type": proto,
            "service": service,
            "flag": flag,
            "src_bytes": int(src_bytes),
            "dst_bytes": int(dst_bytes),
            "land": 1 if (src == dst and sp == dp) else 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": logged_in,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": flow_count,
            "srv_count": flow_count,
            "serror_rate": 1.0 if flag == "S0" else 0.0,
            "srv_serror_rate": 1.0 if flag == "S0" else 0.0,
            "rerror_rate": 1.0 if flag == "REJ" else 0.0,
            "srv_rerror_rate": 1.0 if flag == "REJ" else 0.0,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": min(flow_count, 255),
            "dst_host_srv_count": min(flow_count, 255),
            "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 1.0 if flag == "S0" else 0.0,
            "dst_host_srv_serror_rate": 1.0 if flag == "S0" else 0.0,
            "dst_host_rerror_rate": 1.0 if flag == "REJ" else 0.0,
            "dst_host_srv_rerror_rate": 1.0 if flag == "REJ" else 0.0,
        })
        flow_records.append(row)

    df_flows = pd.DataFrame(flow_records, columns=FEATURE_COLUMNS)
    return df_flows