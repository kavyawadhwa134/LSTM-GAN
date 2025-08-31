#!/usr/bin/env python3
"""
GEANT4-Style Terminal Output System
===================================
Professional terminal output with colors, progress bars, and structured formatting
inspired by GEANT4 simulation output.
"""

import sys
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

class Colors:
    """ANSI color codes for terminal output"""
    # Basic colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Bright colors
    BRIGHT_RED = '\033[1;91m'
    BRIGHT_GREEN = '\033[1;92m'
    BRIGHT_YELLOW = '\033[1;93m'
    BRIGHT_BLUE = '\033[1;94m'
    BRIGHT_MAGENTA = '\033[1;95m'
    BRIGHT_CYAN = '\033[1;96m'
    BRIGHT_WHITE = '\033[1;97m'
    
    # Background colors
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'
    
    # Styles
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    
    # Reset
    RESET = '\033[0m'
    CLEAR = '\033[2J'
    CLEAR_LINE = '\033[2K'

class Geant4Output:
    """GEANT4-style terminal output manager"""
    
    def __init__(self, width: int = 80):
        self.width = width
        self.start_time = time.time()
        self.current_step = 0
        self.total_steps = 0
        
    def clear_screen(self):
        """Clear the terminal screen"""
        print(Colors.CLEAR, end='')
        sys.stdout.flush()
        
    def print_header(self, title: str, version: str = "1.0.0"):
        """Print GEANT4-style header"""
        print(f"{Colors.BRIGHT_CYAN}{'='*self.width}{Colors.RESET}")
        print(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}{title.center(self.width)}{Colors.RESET}")
        print(f"{Colors.CYAN}Version: {version}{Colors.RESET}")
        print(f"{Colors.CYAN}Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{'='*self.width}{Colors.RESET}\n")
        
    def print_section(self, title: str, level: int = 1):
        """Print section header"""
        if level == 1:
            print(f"\n{Colors.BRIGHT_BLUE}{'─'*self.width}{Colors.RESET}")
            print(f"{Colors.BRIGHT_BLUE}{Colors.BOLD} {title}{Colors.RESET}")
            print(f"{Colors.BRIGHT_BLUE}{'─'*self.width}{Colors.RESET}")
        elif level == 2:
            print(f"\n{Colors.BLUE}▶ {title}{Colors.RESET}")
        else:
            print(f"{Colors.CYAN}  • {title}{Colors.RESET}")
            
    def print_info(self, message: str, prefix: str = "INFO"):
        """Print info message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"{Colors.GREEN}[{timestamp}] {Colors.BRIGHT_GREEN}{prefix:8s}{Colors.RESET} {message}")
        
    def print_warning(self, message: str, prefix: str = "WARNING"):
        """Print warning message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"{Colors.YELLOW}[{timestamp}] {Colors.BRIGHT_YELLOW}{prefix:8s}{Colors.RESET} {message}")
        
    def print_error(self, message: str, prefix: str = "ERROR"):
        """Print error message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"{Colors.RED}[{timestamp}] {Colors.BRIGHT_RED}{prefix:8s}{Colors.RESET} {message}")
        
    def print_success(self, message: str, prefix: str = "SUCCESS"):
        """Print success message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"{Colors.GREEN}[{timestamp}] {Colors.BRIGHT_GREEN}{prefix:8s}{Colors.RESET} {message}")
        
    def print_progress_bar(self, current: int, total: int, prefix: str = "Progress", 
                          suffix: str = "", length: int = 50, show_percent: bool = True):
        """Print progress bar"""
        if total == 0:
            percent = 0
        else:
            percent = 100 * (current / float(total))
            
        filled_length = int(length * current // total) if total > 0 else 0
        bar = '█' * filled_length + '░' * (length - filled_length)
        
        # Color based on progress
        if percent < 30:
            color = Colors.RED
        elif percent < 70:
            color = Colors.YELLOW
        else:
            color = Colors.GREEN
            
        if show_percent:
            print(f'\r{Colors.CYAN}{prefix:12s}{Colors.RESET} |{color}{bar}{Colors.RESET}| {percent:6.1f}% {suffix}', end='', flush=True)
        else:
            print(f'\r{Colors.CYAN}{prefix:12s}{Colors.RESET} |{color}{bar}{Colors.RESET}| {current}/{total} {suffix}', end='', flush=True)
            
        if current == total:
            print()  # New line when complete
            
    def print_table(self, headers: List[str], rows: List[List[str]], 
                   title: str = "", border_style: str = "double"):
        """Print formatted table"""
        if not rows:
            return
            
        # Calculate column widths
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Add padding
        col_widths = [w + 2 for w in col_widths]
        total_width = sum(col_widths) + len(headers) + 1
        
        # Print title
        if title:
            print(f"\n{Colors.BRIGHT_WHITE}{Colors.BOLD}{title.center(total_width)}{Colors.RESET}")
            
        # Print top border
        if border_style == "double":
            border_char = "═"
            corner_chars = ["╔", "╗", "╚", "╝", "╠", "╣", "╦", "╩", "╬"]
        else:
            border_char = "─"
            corner_chars = ["┌", "┐", "└", "┘", "├", "┤", "┬", "┴", "┼"]
            
        top_border = corner_chars[0] + border_char * (total_width - 2) + corner_chars[1]
        print(f"{Colors.BRIGHT_CYAN}{top_border}{Colors.RESET}")
        
        # Print header
        header_row = "│"
        for i, (header, width) in enumerate(zip(headers, col_widths)):
            header_row += f" {Colors.BRIGHT_WHITE}{Colors.BOLD}{header:<{width-1}}{Colors.RESET}│"
        print(f"{Colors.BRIGHT_CYAN}{header_row}{Colors.RESET}")
        
        # Print separator
        sep_row = corner_chars[4] + border_char * (total_width - 2) + corner_chars[5]
        print(f"{Colors.BRIGHT_CYAN}{sep_row}{Colors.RESET}")
        
        # Print data rows
        for row in rows:
            data_row = "│"
            for i, (cell, width) in enumerate(zip(row, col_widths)):
                data_row += f" {str(cell):<{width-1}}│"
            print(f"{Colors.BRIGHT_CYAN}{data_row}{Colors.RESET}")
            
        # Print bottom border
        bottom_border = corner_chars[2] + border_char * (total_width - 2) + corner_chars[3]
        print(f"{Colors.BRIGHT_CYAN}{bottom_border}{Colors.RESET}\n")
        
    def print_event_summary(self, event_counts: Dict[str, Dict], total_events: int):
        """Print event summary in GEANT4 style"""
        self.print_section("EVENT SUMMARY", 1)
        
        # Create table data
        headers = ["Event Type", "Count", "Percentage", "Status"]
        rows = []
        
        event_colors = {
            'scattering': Colors.BLUE,
            'absorption': Colors.RED,
            'fission': Colors.YELLOW,
            'leakage': Colors.MAGENTA,
            'capture': Colors.GREEN
        }
        
        for event_type, data in event_counts.items():
            count = data.get('count', 0)
            percentage = data.get('percentage', 0)
            
            # Determine status
            if percentage > 50:
                status = f"{Colors.BRIGHT_GREEN}DOMINANT{Colors.RESET}"
            elif percentage > 20:
                status = f"{Colors.YELLOW}COMMON{Colors.RESET}"
            elif percentage > 5:
                status = f"{Colors.CYAN}MODERATE{Colors.RESET}"
            else:
                status = f"{Colors.RED}RARE{Colors.RESET}"
                
            color = event_colors.get(event_type, Colors.WHITE)
            rows.append([
                f"{color}{event_type.upper()}{Colors.RESET}",
                f"{count:,}",
                f"{percentage:.1f}%",
                status
            ])
            
        self.print_table(headers, rows, "Neutron Event Distribution")
        
    def print_physics_summary(self, physics_data: Dict[str, Any]):
        """Print physics summary"""
        self.print_section("PHYSICS SUMMARY", 1)
        
        headers = ["Property", "Value", "Unit", "Status"]
        rows = []
        
        # Energy statistics
        if 'energy_stats' in physics_data:
            energy = physics_data['energy_stats']
            rows.append([
                "Mean Energy",
                f"{energy.get('mean', 0):.6f}",
                "MeV",
                f"{Colors.GREEN}OK{Colors.RESET}" if energy.get('mean', 0) > 0 else f"{Colors.RED}INVALID{Colors.RESET}"
            ])
            rows.append([
                "Energy Range",
                f"{energy.get('min', 0):.6f} - {energy.get('max', 0):.6f}",
                "MeV",
                f"{Colors.GREEN}OK{Colors.RESET}" if energy.get('max', 0) > energy.get('min', 0) else f"{Colors.RED}INVALID{Colors.RESET}"
            ])
            
        # Position statistics
        if 'position_stats' in physics_data:
            pos = physics_data['position_stats']
            rows.append([
                "Position Range",
                f"({pos.get('x_range', 0):.3f}, {pos.get('y_range', 0):.3f}, {pos.get('z_range', 0):.3f})",
                "cm",
                f"{Colors.GREEN}OK{Colors.RESET}"
            ])
            
        self.print_table(headers, rows, "Physics Validation")
        
    def print_training_progress(self, iteration: int, total_iterations: int, 
                              g_loss: float, d_loss: float, accuracy: float,
                              event_loss: float = 0.0, physics_loss: float = 0.0):
        """Print training progress in GEANT4 style"""
        elapsed = time.time() - self.start_time
        
        # Clear line and print progress
        print(f"\r{Colors.CLEAR_LINE}", end='')
        
        # Progress bar
        self.print_progress_bar(iteration, total_iterations, "Training", 
                              f"({iteration}/{total_iterations})", 40)
        
        # Loss information
        print(f"{Colors.CYAN}Generator Loss:{Colors.RESET} {Colors.YELLOW}{g_loss:.6f}{Colors.RESET} | "
              f"{Colors.CYAN}Discriminator Loss:{Colors.RESET} {Colors.YELLOW}{d_loss:.6f}{Colors.RESET} | "
              f"{Colors.CYAN}Accuracy:{Colors.RESET} {Colors.GREEN}{accuracy:.1f}%{Colors.RESET}")
        
        if event_loss > 0:
            print(f"{Colors.CYAN}Event Loss:{Colors.RESET} {Colors.MAGENTA}{event_loss:.6f}{Colors.RESET} | "
                  f"{Colors.CYAN}Physics Loss:{Colors.RESET} {Colors.BLUE}{physics_loss:.6f}{Colors.RESET}")
        
        # Time information
        eta = (elapsed / iteration) * (total_iterations - iteration) if iteration > 0 else 0
        print(f"{Colors.CYAN}Elapsed:{Colors.RESET} {elapsed:.1f}s | "
              f"{Colors.CYAN}ETA:{Colors.RESET} {eta:.1f}s")
        
    def print_final_summary(self, total_time: float, total_events: int, 
                          accuracy: float, model_info: Dict[str, Any]):
        """Print final summary"""
        self.print_section("SIMULATION COMPLETED", 1)
        
        print(f"{Colors.BRIGHT_GREEN}✓ Simulation completed successfully!{Colors.RESET}\n")
        
        # Summary table
        headers = ["Metric", "Value", "Status"]
        rows = [
            ["Total Runtime", f"{total_time:.2f} seconds", f"{Colors.GREEN}COMPLETED{Colors.RESET}"],
            ["Total Events", f"{total_events:,}", f"{Colors.GREEN}GENERATED{Colors.RESET}"],
            ["Model Accuracy", f"{accuracy:.1f}%", f"{Colors.GREEN if accuracy > 70 else Colors.YELLOW}GOOD{Colors.RESET}"],
            ["Model Type", model_info.get('type', 'LSTM-WGAN'), f"{Colors.CYAN}ACTIVE{Colors.RESET}"],
            ["Device", model_info.get('device', 'CPU'), f"{Colors.BLUE}USED{Colors.RESET}"]
        ]
        
        self.print_table(headers, rows, "Final Results")
        
        # Performance metrics
        if total_time > 0:
            events_per_second = total_events / total_time
            print(f"{Colors.CYAN}Performance:{Colors.RESET} {events_per_second:.0f} events/second")
        
        print(f"\n{Colors.BRIGHT_CYAN}{'='*self.width}{Colors.RESET}")
        print(f"{Colors.BRIGHT_WHITE}Simulation finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{'='*self.width}{Colors.RESET}")

# Example usage and testing
if __name__ == "__main__":
    # Create GEANT4-style output manager
    g4_output = Geant4Output()
    
    # Clear screen and print header
    g4_output.clear_screen()
    g4_output.print_header("LSTM-WGAN Neutron Physics Simulation", "2.0.0")
    
    # Print sections
    g4_output.print_section("INITIALIZATION", 1)
    g4_output.print_info("Loading neutron data from Sheet.csv")
    g4_output.print_info("Initializing LSTM-WGAN model")
    g4_output.print_success("Model initialized successfully")
    
    # Print progress bar
    g4_output.print_section("TRAINING PROGRESS", 1)
    for i in range(101):
        g4_output.print_progress_bar(i, 100, "Training", f"Iteration {i}")
        time.sleep(0.05)
    
    # Print event summary
    event_counts = {
        'scattering': {'count': 15677, 'percentage': 31.4},
        'absorption': {'count': 96, 'percentage': 0.2},
        'fission': {'count': 2852, 'percentage': 5.7},
        'leakage': {'count': 1, 'percentage': 0.0},
        'capture': {'count': 31374, 'percentage': 62.7}
    }
    g4_output.print_event_summary(event_counts, 50000)
    
    # Print physics summary
    physics_data = {
        'energy_stats': {'mean': 0.025, 'min': 0.001, 'max': 2.0},
        'position_stats': {'x_range': 10.0, 'y_range': 10.0, 'z_range': 10.0}
    }
    g4_output.print_physics_summary(physics_data)
    
    # Print final summary
    model_info = {'type': 'LSTM-WGAN', 'device': 'CUDA'}
    g4_output.print_final_summary(120.5, 50000, 72.9, model_info)
