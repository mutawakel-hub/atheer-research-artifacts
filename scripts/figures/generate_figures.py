"""
Generate all figures for the Atheer v2 paper.
Saves PNGs to /home/z/my-project/scripts/figures/
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm

# Register fonts (per skill rules)
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

FIG_DIR = "/home/z/my-project/scripts/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Color palette (IEEE-friendly, low-saturation)
COLOR_S1 = '#C44536'   # muted red for S1 (public internet)
COLOR_S2 = '#2E5C8A'   # muted blue for S2 (mobile data)
COLOR_ACCENT = '#5A8F5A'  # green accent
COLOR_LIGHT = '#F5F5F0'
COLOR_TEXT = '#1A1A1A'


# ============================================================
# Load simulation results
# ============================================================
def load_results():
    with open("/home/z/my-project/scripts/sim_results/aggregated.json") as f:
        return json.load(f)


# ============================================================
# Figure 6: Transaction Success Rate
# ============================================================
def plot_success_rate(results):
    tps_levels = [5, 25, 50, 100, 250, 500]
    s1_means = [results["S1"][str(t)]["success_rate_mean"] for t in tps_levels]
    s1_cis = [results["S1"][str(t)]["success_rate_ci"] for t in tps_levels]
    s2_means = [results["S2"][str(t)]["success_rate_mean"] for t in tps_levels]
    s2_cis = [results["S2"][str(t)]["success_rate_ci"] for t in tps_levels]

    fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)

    ax.errorbar(tps_levels, s1_means, yerr=s1_cis, marker='s',
                color=COLOR_S1, label='S1: Public Internet',
                capsize=3, linewidth=1.5, markersize=6)
    ax.errorbar(tps_levels, s2_means, yerr=s2_cis, marker='o',
                color=COLOR_S2, label='S2: Mobile Data (Partner-Subsidized)',
                capsize=3, linewidth=1.5, markersize=6)

    ax.set_xlabel('Offered Load (TPS)', fontsize=10)
    ax.set_ylabel('Transaction Success Rate (%)', fontsize=10)
    ax.set_title('Transaction Success Rate vs. Load', fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(tps_levels)
    ax.set_xticklabels([str(t) for t in tps_levels])
    ax.set_ylim(40, 102)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.95)
    ax.tick_params(labelsize=9)

    fig.savefig(os.path.join(FIG_DIR, "fig6_success_rate.png"), dpi=200)
    plt.close(fig)
    print("Saved: fig6_success_rate.png")


# ============================================================
# Figure 7: P95 End-to-End Latency
# ============================================================
def plot_p95_latency(results):
    tps_levels = [5, 25, 50, 100, 250, 500]
    s1_means = [results["S1"][str(t)]["p95_latency_ms_mean"] / 1000 for t in tps_levels]
    s1_cis = [results["S1"][str(t)]["p95_latency_ms_ci"] / 1000 for t in tps_levels]
    s2_means = [results["S2"][str(t)]["p95_latency_ms_mean"] / 1000 for t in tps_levels]
    s2_cis = [results["S2"][str(t)]["p95_latency_ms_ci"] / 1000 for t in tps_levels]

    fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)

    ax.errorbar(tps_levels, s1_means, yerr=s1_cis, marker='s',
                color=COLOR_S1, label='S1: Public Internet',
                capsize=3, linewidth=1.5, markersize=6)
    ax.errorbar(tps_levels, s2_means, yerr=s2_cis, marker='o',
                color=COLOR_S2, label='S2: Mobile Data (Partner-Subsidized)',
                capsize=3, linewidth=1.5, markersize=6)

    ax.set_xlabel('Offered Load (TPS)', fontsize=10)
    ax.set_ylabel('P95 End-to-End Latency (s)', fontsize=10)
    ax.set_title('P95 Latency vs. Load', fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(tps_levels)
    ax.set_xticklabels([str(t) for t in tps_levels])
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax.tick_params(labelsize=9)

    fig.savefig(os.path.join(FIG_DIR, "fig7_p95_latency.png"), dpi=200)
    plt.close(fig)
    print("Saved: fig7_p95_latency.png")


# ============================================================
# Figure 1: 4-tier Architecture Diagram (matplotlib-based, vector-style)
# ============================================================
def plot_architecture():
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Layer boxes
    layers = [
        ("Edge Layer\n(SDK + SoftPOS)", 0.5, 4.5, '#E8F0F8', '#2E5C8A'),
        ("Network Layer\n(Mobile Data Routing)", 0.5, 3.0, '#FFF4E8', '#C44536'),
        ("Switch Layer\n(Atheer Gateway)", 0.5, 1.5, '#E8F5E8', '#5A8F5A'),
        ("Integration Layer\n(HSM + Banking Ledger)", 0.5, 0.0, '#F0E8F0', '#7A4F8A'),
    ]
    for label, x, y, fill, edge in layers:
        rect = plt.Rectangle((x, y), 9, 1.3, facecolor=fill, edgecolor=edge,
                              linewidth=1.8, zorder=1)
        ax.add_patch(rect)
        ax.text(x + 4.5, y + 0.65, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color=COLOR_TEXT, zorder=2)

    # Right-side partner wallet callout
    rect = plt.Rectangle((5.5, 2.85), 4.3, 1.6, facecolor='none',
                          edgecolor='#888888', linewidth=1, linestyle='--', zorder=1)
    ax.add_patch(rect)
    ax.text(7.65, 4.25, 'Partner Wallet\n(B2B Billing)', ha='center', va='center',
            fontsize=8, style='italic', color='#555555', zorder=2)

    # Down arrows
    arrow_props = dict(arrowstyle='->', lw=1.6, color='#444444')
    for x in [3.0, 7.0]:
        ax.annotate('', xy=(x, 3.0), xytext=(x, 4.5), arrowprops=arrow_props)
        ax.annotate('', xy=(x, 1.5), xytext=(x, 3.0), arrowprops=arrow_props)
        ax.annotate('', xy=(x, 0.0), xytext=(x, 1.5), arrowprops=arrow_props)

    # Subsidy arrow (right)
    ax.annotate('', xy=(7.65, 3.65), xytext=(7.65, 4.4),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='#888888', linestyle=':'))
    ax.text(8.2, 4.0, 'subsidy', fontsize=7, color='#888888', style='italic')

    ax.set_title('"Atheer" 4-Tier Architecture (Revised)', fontsize=12, fontweight='bold', pad=10)
    fig.savefig(os.path.join(FIG_DIR, "fig1_architecture.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig1_architecture.png")


# ============================================================
# Figure 3: Interaction Architecture
# ============================================================
def plot_interaction():
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Entities
    entities = [
        ("Customer\nDevice\n(SDK)", 1, 3.5, '#E8F0F8', '#2E5C8A'),
        ("Merchant\nSoftPOS", 3.5, 3.5, '#FFF4E8', '#C44536'),
        ("Atheer\nGateway", 6.5, 3.5, '#E8F5E8', '#5A8F5A'),
        ("Core\nBanking", 9, 3.5, '#F0E8F0', '#7A4F8A'),
        ("Partner\nWallet", 6.5, 1, '#FFF8E1', '#888888'),
    ]
    for label, x, y, fill, edge in entities:
        rect = plt.Rectangle((x-0.7, y-0.55), 1.4, 1.1, facecolor=fill,
                              edgecolor=edge, linewidth=1.6)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=8.5, fontweight='bold')

    # Arrows with labels
    arrows = [
        ((1.7, 3.5), (2.8, 3.5), 'NFC + APDU', 'solid'),
        ((4.2, 3.5), (5.8, 3.5), 'Mobile Data\n(TLS 1.3)', 'solid'),
        ((7.2, 3.5), (8.3, 3.5), 'Adapter\nAPI', 'solid'),
        ((6.5, 2.95), (6.5, 1.55), 'B2B\nBilling', 'dashed'),
    ]
    for (x1, y1), (x2, y2), lbl, style in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.3,
                                    color='#444444',
                                    linestyle=style if style == 'dashed' else 'solid'))
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 + 0.25
        ax.text(mid_x, mid_y, lbl, ha='center', va='bottom',
                fontsize=7.5, color='#333333', style='italic')

    ax.set_title('Interaction Among Atheer Gateway, SoftPOS, SDK, Banking, and Partner Wallet',
                 fontsize=10, fontweight='bold', pad=10)
    fig.savefig(os.path.join(FIG_DIR, "fig3_interaction.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig3_interaction.png")


# ============================================================
# Figure 4: State Diagram of Armed Session
# ============================================================
def plot_state_diagram():
    fig, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    states = [
        ("Idle", 1, 2, '#E0E0E0', '#555555'),
        ("Biometric\nAuth", 3, 2, '#E8F0F8', '#2E5C8A'),
        ("Armed\nSession\n(60s)", 5.5, 2, '#FFF4E8', '#C44536'),
        ("APDU\nTransmitted", 8, 2, '#E8F5E8', '#5A8F5A'),
        ("Expired", 5.5, 0.5, '#F8E8E8', '#888888'),
        ("Rejected", 5.5, 3.5, '#F8E8E8', '#888888'),
    ]
    for label, x, y, fill, edge in states:
        ellipse = plt.Circle((x, y), 0.65, facecolor=fill, edgecolor=edge, linewidth=1.6)
        ax.add_patch(ellipse)
        ax.text(x, y, label, ha='center', va='center', fontsize=8.5, fontweight='bold')

    # Transitions
    transitions = [
        ((1.65, 2), (2.35, 2), 'auth\nrequest'),
        ((3.65, 2), (4.85, 2), 'success'),
        ((6.15, 2), (7.35, 2), 'pay\ntrigger'),
        ((5.5, 1.35), (5.5, 1.15), 'timeout'),
        ((5.5, 2.65), (5.5, 2.85), 'invalid'),
    ]
    for (x1, y1), (x2, y2), lbl in transitions:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.2, color='#444444'))
        ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, lbl,
                fontsize=7, color='#333333', style='italic')

    ax.set_title('State Diagram: Armed Session Lifecycle',
                 fontsize=11, fontweight='bold', pad=10)
    fig.savefig(os.path.join(FIG_DIR, "fig4_state_diagram.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig4_state_diagram.png")


# ============================================================
# Figure 5: Secure Charge Request Packet Structure (180 bytes)
# ============================================================
def plot_packet_structure():
    fig, ax = plt.subplots(figsize=(7, 2.3), constrained_layout=True)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Field breakdown (cumulative byte positions for visualization)
    fields = [
        ("Routing\nHeader", 32, '#E8F0F8', '#2E5C8A'),
        ("LUK\nToken", 32, '#FFF4E8', '#C44536'),
        ("ATC", 4, '#E8F5E8', '#5A8F5A'),
        ("Amount", 8, '#F0E8F0', '#7A4F8A'),
        ("Nonce", 16, '#FFF8E1', '#888888'),
        ("Time-\nstamp", 8, '#E8F0F8', '#2E5C8A'),
        ("ECDSA\nSig", 64, '#FFF4E8', '#C44536'),
        ("AES\nIV+Tag", 28, '#E8F5E8', '#5A8F5A'),
    ]
    # Scale: each unit = 10 bytes
    x = 0
    for label, size, fill, edge in fields:
        w = size / 10
        rect = plt.Rectangle((x, 1.2), w, 1.6, facecolor=fill, edgecolor=edge, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, 2.0, label, ha='center', va='center',
                fontsize=7.5, fontweight='bold')
        ax.text(x + w/2, 0.85, f"{size}B", ha='center', va='center',
                fontsize=7, color='#555555')
        x += w

    ax.text(9, 3.6, 'Total: 192 bytes (before compression) -> 180 bytes (after header compression)',
            ha='center', fontsize=8.5, fontweight='bold', color=COLOR_TEXT)
    ax.set_title('Secure Charge Request Packet Structure (180 Bytes Effective)',
                 fontsize=10, fontweight='bold', pad=2)
    fig.savefig(os.path.join(FIG_DIR, "fig5_packet.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig5_packet.png")


# ============================================================
# Figure 2: Layered system breakdown (Edge + Switch)
# ============================================================
def plot_layered_system():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)

    # (a) Edge Layer
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 6)
    ax1.axis('off')
    edge_modules = [
        ("Cryptographic\nKey Manager", 1.5, 4.5),
        ("Secure\nToken Vault", 4.5, 4.5),
        ("HCE\nEngine", 1.5, 2.8),
        ("Network\nRouting Unit", 4.5, 2.8),
        ("Encrypted\nLocal Store", 3.0, 1.1),
    ]
    for label, x, y in edge_modules:
        rect = plt.Rectangle((x-0.85, y-0.45), 1.7, 0.9,
                              facecolor='#E8F0F8', edgecolor='#2E5C8A', linewidth=1.4)
        ax1.add_patch(rect)
        ax1.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')
    ax1.set_title('(a) Edge Layer', fontsize=10, fontweight='bold')

    # (b) Cloud Switch Layer
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    switch_modules = [
        ("Ingress API\n(TLS 1.3)", 1.5, 4.7),
        ("ECDSA\nVerifier", 4.5, 4.7),
        ("Zero-Trust\nIdentity Lookup", 1.5, 3.2),
        ("Nonce Cache\n(Replay Guard)", 4.5, 3.2),
        ("Adapter\nRouter", 1.5, 1.7),
        ("Token Status\nRegistry", 4.5, 1.7),
        ("Banking HSM\nConnector", 3.0, 0.4),
    ]
    for label, x, y in switch_modules:
        rect = plt.Rectangle((x-0.85, y-0.4), 1.7, 0.8,
                              facecolor='#E8F5E8', edgecolor='#5A8F5A', linewidth=1.4)
        ax2.add_patch(rect)
        ax2.text(x, y, label, ha='center', va='center', fontsize=7.5, fontweight='bold')
    ax2.set_title('(b) Cloud Switch Layer', fontsize=10, fontweight='bold')

    fig.savefig(os.path.join(FIG_DIR, "fig2_layered.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: fig2_layered.png")


# ============================================================
# Figure 8: Cost Model Sensitivity (additional)
# ============================================================
def plot_cost_model():
    fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)

    # Daily transactions range
    n_txn = np.logspace(3, 6, 50)
    payload_bytes = 180
    prices_per_mb = [0.50, 1.00, 2.00]
    avg_amount_usd = 5.0
    mdr_rate = 0.01  # 1%

    for price in prices_per_mb:
        daily_data_mb = (n_txn * payload_bytes) / (1024 * 1024)
        daily_cost = daily_data_mb * price
        daily_revenue = n_txn * avg_amount_usd * mdr_rate
        ratio = (daily_cost / daily_revenue) * 100
        ax.plot(n_txn, ratio, label=f"${price:.2f}/MB", linewidth=1.8)

    ax.set_xscale('log')
    ax.set_xlabel('Daily Transactions (log scale)', fontsize=10)
    ax.set_ylabel('Data Cost / MDR Revenue (%)', fontsize=10)
    ax.set_title('Cost Sensitivity: Data Cost vs. MDR Revenue', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(title='Mobile Data Price', fontsize=8, title_fontsize=9)
    ax.tick_params(labelsize=9)
    ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.6, label='1% threshold')
    ax.text(1e3, 1.1, '1% threshold', fontsize=7, color='red')

    fig.savefig(os.path.join(FIG_DIR, "fig8_cost_model.png"), dpi=200)
    plt.close(fig)
    print("Saved: fig8_cost_model.png")


if __name__ == "__main__":
    results = load_results()
    plot_architecture()
    plot_layered_system()
    plot_interaction()
    plot_state_diagram()
    plot_packet_structure()
    plot_success_rate(results)
    plot_p95_latency(results)
    plot_cost_model()
    print("\nAll figures generated successfully.")
