"""Generate a test PCAP containing normal HTTP connections and SYN flood attacks."""

from pathlib import Path
from scapy.all import IP, TCP, wrpcap

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PCAP_PATH = DATA_DIR / "sample_traffic.pcap"


def generate_pcap():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pkts = []

    # 1. Benign HTTP Handshake and Data Transfer
    client_ip = "192.168.1.105"
    server_ip = "192.168.1.1"

    # SYN -> SYN-ACK -> ACK
    pkts.append(IP(src=client_ip, dst=server_ip) / TCP(sport=54321, dport=80, flags="S"))
    pkts.append(IP(src=server_ip, dst=client_ip) / TCP(sport=80, dport=54321, flags="SA"))
    pkts.append(IP(src=client_ip, dst=server_ip) / TCP(sport=54321, dport=80, flags="A"))
    # HTTP payload packet
    pkts.append(
        IP(src=client_ip, dst=server_ip)
        / TCP(sport=54321, dport=80, flags="PA")
        / b"GET /index.html HTTP/1.1\r\n\r\n"
    )

    # 2. Malicious SYN Flood (PortScan / DoS)
    attacker_ip = "10.0.0.99"
    for target_port in [80, 443, 22, 21, 8080, 3306]:
        np_sport = 40000 + target_port
        pkts.append(
            IP(src=attacker_ip, dst=server_ip)
            / TCP(sport=np_sport, dport=target_port, flags="S")
        )

    wrpcap(str(PCAP_PATH), pkts)
    print(f"Generated sample PCAP with {len(pkts)} packets at: {PCAP_PATH}")


if __name__ == "__main__":
    generate_pcap()