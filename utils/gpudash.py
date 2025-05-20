############################################################
#
# gpudash.py
#
# A Colorful GPU Dashboard for ANSI terminals.
#
# Luke Sheneman
# sheneman@uidaho.edu
# Institute for Interdisciplinary Data Scienace (IIDS)
#
# Works great on MacOS Terminal.app
#
# Configuration loaded from gpudash.conf
#
############################################################

import subprocess
import threading
import time
import datetime
import re # For parsing power draw
import json
import sys
import os # For checking config file path

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from rich.padding import Padding # For spacing between host panels

# --- Configuration File ---
CONFIG_FILE_PATH = "gpudash.conf"

# --- Configuration (will be loaded from CONFIG_FILE_PATH) ---
HOSTS = []
SSH_USER = ""
SSH_TIMEOUT = 7
REFRESH_INTERVAL = 5
# BAR_WIDTH is now dynamic

# --- Global Data Store ---
gpu_data = {}
status_messages = {}

def load_configuration():
    """Loads HOSTS and SSH_USER from gpudash.conf."""
    global HOSTS, SSH_USER

    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Error: Configuration file '{CONFIG_FILE_PATH}' not found.", file=sys.stderr)
        print("Please create it with 'HOSTS' (list of strings) and 'SSH_USER' (string).", file=sys.stderr)
        sys.exit(1)

    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{CONFIG_FILE_PATH}'. Please check its format.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading configuration file '{CONFIG_FILE_PATH}': {e}", file=sys.stderr)
        sys.exit(1)

    # Validate and load HOSTS
    if "HOSTS" not in config:
        print(f"Error: 'HOSTS' key missing in '{CONFIG_FILE_PATH}'.", file=sys.stderr)
        sys.exit(1)
    if not isinstance(config["HOSTS"], list) or not all(isinstance(h, str) for h in config["HOSTS"]):
        print(f"Error: 'HOSTS' in '{CONFIG_FILE_PATH}' must be a list of strings.", file=sys.stderr)
        sys.exit(1)
    HOSTS = config["HOSTS"]

    # Validate and load SSH_USER
    if "SSH_USER" not in config:
        print(f"Error: 'SSH_USER' key missing in '{CONFIG_FILE_PATH}'.", file=sys.stderr)
        sys.exit(1)
    if not isinstance(config["SSH_USER"], str):
        print(f"Error: 'SSH_USER' in '{CONFIG_FILE_PATH}' must be a string.", file=sys.stderr)
        sys.exit(1)
    SSH_USER = config["SSH_USER"]

    if not SSH_USER:
        print(f"Error: 'SSH_USER' in '{CONFIG_FILE_PATH}' cannot be empty.", file=sys.stderr)
        sys.exit(1)

    if not HOSTS:
        print(f"Warning: 'HOSTS' list in '{CONFIG_FILE_PATH}' is empty. No hosts to monitor.", file=sys.stderr)
        # The program will run but show no hosts.
        # Consider if sys.exit(1) is more appropriate here.


def get_gpu_util_from_host(hostname):
    global gpu_data, status_messages, SSH_USER, SSH_TIMEOUT # Ensure SSH_USER and SSH_TIMEOUT are accessible
    cmd = [
        "ssh",
        "-o", f"ConnectTimeout={SSH_TIMEOUT}",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new", # Consider security implications if hosts change IPs often
        f"{SSH_USER}@{hostname}",
        "nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SSH_TIMEOUT + 2)
        if proc.returncode == 0:
            host_gpus_data = []
            lines = proc.stdout.strip().split('\n')
            if not lines or not lines[0].strip(): # Handle empty output even if command succeeds
                status_messages[hostname] = "No GPUs found or no data"
                gpu_data[hostname] = []
                return

            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 3:
                    try:
                        util = int(parts[0])
                        power_match = re.match(r"([\d\.]+)", parts[1]) # Handle potential "N/A" or other non-numeric
                        power = float(power_match.group(1)) if power_match else 0.0
                        temp = int(parts[2])
                        host_gpus_data.append({"util": util, "power": power, "temp": temp})
                    except ValueError:
                        # Could log this specific parsing error if needed
                        pass # Skip malformed line
            
            if not host_gpus_data and proc.stdout.strip(): # Output was there but not parsed
                 status_messages[hostname] = "No GPU data parsed / bad format"
                 gpu_data[hostname] = []
            elif not host_gpus_data: # No valid GPU lines at all
                status_messages[hostname] = "No GPUs reported"
                gpu_data[hostname] = []
            else:
                gpu_data[hostname] = host_gpus_data
                if hostname in status_messages: del status_messages[hostname] # Clear previous error
        else:
            error_msg = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"nvidia-smi error (code {proc.returncode})"
            status_messages[hostname] = error_msg[:70] # Truncate long messages
            gpu_data[hostname] = []
    except subprocess.TimeoutExpired:
        status_messages[hostname] = "SSH/Cmd Timeout"
        gpu_data[hostname] = []
    except Exception as e:
        status_messages[hostname] = f"SSH Error: {str(e)[:50]}" # Truncate
        gpu_data[hostname] = []

def fetch_all_data():
    threads = []
    for host in HOSTS: # Uses the global HOSTS list loaded from config
        thread = threading.Thread(target=get_gpu_util_from_host, args=(host,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

def generate_host_panel(hostname: str, width_available_for_panel: int) -> Panel:
    """Generates a Panel for a single host's GPU data, with dynamic bar width."""
    
    panel_horizontal_overhead = 2 + 2 
    table_area_width = width_available_for_panel - panel_horizontal_overhead

    gpu_col_content_w = 5
    util_col_content_w = 6
    pwr_col_content_w = 7
    temp_col_content_w = 6
    table_cell_h_padding_per_cell_total = 2 

    fixed_cols_total_rendered_width = (
        (gpu_col_content_w + table_cell_h_padding_per_cell_total) +
        (util_col_content_w + table_cell_h_padding_per_cell_total) +
        (pwr_col_content_w + table_cell_h_padding_per_cell_total) +
        (temp_col_content_w + table_cell_h_padding_per_cell_total)
    )

    remaining_width_for_bar_cell_rendered = table_area_width - fixed_cols_total_rendered_width
    bar_col_content_target_w = max(3, remaining_width_for_bar_cell_rendered - table_cell_h_padding_per_cell_total)
    dynamic_bar_chars_for_pipe = max(1, bar_col_content_target_w - 2)

    host_table = Table(
        title=None, box=None, show_header=True,
        header_style="bold magenta", padding=(0,1,0,1)
    )
    host_table.add_column("GPU", style="dim white", width=gpu_col_content_w)
    host_table.add_column("Util", justify="right", width=util_col_content_w)
    host_table.add_column("Pwr", justify="right", width=pwr_col_content_w)
    host_table.add_column("Temp", justify="right", width=temp_col_content_w)
    host_table.add_column("Bar", width=bar_col_content_target_w)

    content: object

    if hostname in status_messages:
        content = Padding(Text(status_messages[hostname], style="red"), (1,2))
    elif hostname in gpu_data and gpu_data[hostname]:
        for i, data in enumerate(gpu_data[hostname]):
            util = data.get("util", 0)
            power = data.get("power", 0.0)
            temp = data.get("temp", 0)

            if util > 85: util_style_color = "red"
            elif util > 60: util_style_color = "yellow"
            else: util_style_color = "green"

            bar_fill_count = int((util / 100) * dynamic_bar_chars_for_pipe)
            bar_text = Text()
            bar_text.append("[")
            bar_text.append("|" * bar_fill_count, style=f"bold {util_style_color}")
            bar_text.append(" " * (dynamic_bar_chars_for_pipe - bar_fill_count))
            bar_text.append("]")

            host_table.add_row(
                f"{i}", Text(f"{util}%", style=util_style_color),
                f"{power:.1f}W", f"{temp}°C", bar_text
            )
        content = host_table
    elif hostname in gpu_data and not gpu_data[hostname] and hostname not in status_messages:
        # This case covers when a host is reachable, nvidia-smi runs, but reports no GPUs.
        content = Padding(Text("No GPUs reported for this host.", style="dim white"), (1,2))
    else: # Fallback, e.g., initial state before first fetch for this host
        content = Padding(Text("Fetching data...", style="dim white"), (1,2))

    return Panel(
        content, title=f"[bold cyan]{hostname}[/]",
        border_style="green", padding=(0,1)
    )

def generate_layout(console_width: int) -> Layout:
    width_for_host_panels_group = console_width - 2

    # HOSTS is now loaded from config
    hosts_in_order = HOSTS 
    host_panels = [generate_host_panel(h, width_for_host_panels_group) for h in hosts_in_order]
    
    spaced_host_panels = []
    for i, panel in enumerate(host_panels):
        spaced_host_panels.append(panel)
        if i < len(host_panels) - 1:
            spaced_host_panels.append(Padding("", (1,0,0,0))) 

    main_content_group = Group(*spaced_host_panels) if spaced_host_panels else Text("No hosts to display.", justify="center")


    total_util_sum = 0
    total_gpus_count = 0
    total_power_sum = 0.0
    for hostname in HOSTS: # Iterate over configured hosts
        if hostname in gpu_data and gpu_data[hostname]:
            for gpu_info in gpu_data[hostname]:
                total_util_sum += gpu_info.get("util", 0)
                total_power_sum += gpu_info.get("power", 0.0)
                total_gpus_count += 1
    average_cluster_util = (total_util_sum / total_gpus_count) if total_gpus_count > 0 else 0.0

    summary_line1_markup = f"Total Avg. GPU Utilization: [bold yellow]{average_cluster_util:.1f}%[/]"
    summary_line2_markup = f"Total Cluster Power Draw: [bold yellow]{total_power_sum:.1f}W[/]"
    summary_content = Text.from_markup(
        f"{summary_line1_markup}\n{summary_line2_markup}", justify="center"
    )
    summary_panel = Panel(
        summary_content, title="[bold]Cluster Summary[/]",
        border_style="green", padding=(1, 2)
    )

    header_panel_content = Text("Inference Cluster Utilization Dashboard", justify="center", style="bold white")
    header_panel = Panel(
        header_panel_content, style="on royal_blue1",
        border_style="green", padding=(0,1)
    )

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer_text_content = f"Last update: {now} | Refresh: {REFRESH_INTERVAL}s | Press Ctrl+C to quit"
    footer_panel_content = Text.from_markup(footer_text_content, justify="center")
    footer_panel = Panel(
        footer_panel_content, style="on default",
        border_style="green", padding=(0,1)
    )

    summary_panel_height = 6
    layout = Layout(name="root")
    layout.split_column(
        Layout(header_panel, name="header", size=3),
        Layout(Panel(main_content_group, border_style="green", title=None, padding=0), name="main_scrollable_area"),
        Layout(summary_panel, name="summary", size=summary_panel_height),
        Layout(footer_panel, name="footer", size=3)
    )
    return layout

def main_rich():
    console = Console()
    
    # HOSTS is now populated by load_configuration() before this runs
    # Initialize gpu_data for all configured hosts
    for host_init in HOSTS:
        if host_init not in gpu_data: 
            gpu_data[host_init] = [] # Initialize as empty list, signifies data not yet fetched or no GPUs
            # status_messages[host_init] = "Initializing..." # Optional: initial status

    fetch_all_data() 
    initial_layout = generate_layout(console.width)

    with Live(initial_layout, console=console, screen=True, refresh_per_second=4, vertical_overflow="visible") as live:
        try:
            while True:
                time.sleep(REFRESH_INTERVAL)
                fetch_all_data()
                live.update(generate_layout(live.console.width))
        except KeyboardInterrupt:
            console.print("Exiting dashboard...", style="yellow")
        finally:
            pass

if __name__ == "__main__":
    load_configuration() # Load config before anything else
    if not HOSTS:
        print("No hosts configured in gpudash.conf. Exiting.", file=sys.stderr)
        sys.exit(0) # Not an error, but nothing to do.
    main_rich()
