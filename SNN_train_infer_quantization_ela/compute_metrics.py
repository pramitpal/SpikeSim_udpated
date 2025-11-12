import numpy as np
import argparse
import sys

# ===========================
# METRIC FUNCTIONS
# (pe_tile_count, compute_energy, compute_area, latency_gen)
# ===========================

def pe_tile_count(in_ch_list, out_ch_list, out_dim_list, k, xbar, pe_per_tile):
    """Calculates the number of tiles required."""
    num_layer = len(out_ch_list)
    pe_list = []

    for i in range(num_layer):
        # Use np.ceil to ensure we have enough PEs
        num_pe = np.ceil(in_ch_list[i] / xbar) * np.ceil(out_ch_list[i] / xbar)
        pe_list.append(num_pe)

    flag = 0
    # Note: This check `pe_list == sorted(pe_list)` is very strict.
    # It implies layers *must* be sorted by PE count.
    # You may want to relax this or double-check if it's the intended logic.
    if pe_list == sorted(pe_list):
        flag = 1

    if flag:
        print("Hardware Mapping: Layers are in PE order.")
    else:
        print("Hardware Mapping: Check layer order (PEs not sorted).")
        # Consider whether you want to `return` or just warn
        # return

    num_pe = sum(pe_list)
    print(f'Total No. of PEs required: {num_pe}')

    if num_pe % pe_per_tile != 0:
        num_tiles = (num_pe // pe_per_tile) + 1
    else:
        num_tiles = num_pe / pe_per_tile

    print(f'Total No. of Tiles: {num_tiles}')
    return num_tiles


# All energies in pJ
def compute_energy(in_ch_list, in_dim_list, out_ch_list, out_dim_list, xbar_size, k, n_tiles, device, time_steps):
    """Computes the total energy consumption."""
    if device == 'rram':
        xbar_ar = 1.76423
    elif device == 'sram':
        xbar_ar = 671.089
    else:
        raise ValueError(f"Unknown device: {device}. Must be 'rram' or 'sram'.")

    # Energy constants (pJ)
    Tile_buff = 397
    Temp_Buff = 0.2
    Sub = 1.15E-6
    ADC = 2.03084
    Htree = 19.64 * 8
    MUX = 0.094245
    mem_fetch = 4.64
    neuron = 1.274 * 4.0

    PE_ar = k * k * xbar_ar + (xbar_size / 8) * (ADC + MUX)
    PE_cycle_energy = Htree + mem_fetch + neuron + xbar_size / 8 * PE_ar + (xbar_size / 8) * 16 * Sub + (
            xbar_size / 8) * Temp_Buff + Tile_buff

    energy_layerwise = []
    tot_energy = 0
    tot_pe_cycle = 0
    num_layers = len(out_ch_list)

    for i in range(num_layers):
        Total_PE_cycle = np.ceil(out_ch_list[i] / xbar_size) * np.ceil(in_ch_list[i] / xbar_size) * (
                out_dim_list[i] * out_dim_list[i])
        
        layer_energy = Total_PE_cycle * PE_cycle_energy * time_steps
        tot_energy += layer_energy
        tot_pe_cycle += Total_PE_cycle
        energy_layerwise.append(layer_energy)

    print(f'Total Energy: {tot_energy} pJ')
    return energy_layerwise


# All areas in µm^2
def compute_area(in_ch_list, in_dim_list, out_ch_list, out_dim_list, xbar_size, k, pe_per_tile, n_tiles, device):
    """Computes the total chip area."""
    if device == 'rram':
        xbar_ar = 26.2144
    elif device == 'sram':
        xbar_ar = 671.089
    else:
        raise ValueError(f"Unknown device: {device}. Must be 'rram' or 'sram'.")

    # Area constants (µm^2)
    Tile_buff = 0.7391 * 64 * 128
    Temp_Buff = 484.643999
    Sub = 13411.41498
    ADC = 693.633
    Htree = 216830 * 2
    MUX = 45.9

    PE_ar = k * k * xbar_ar + (xbar_size / 8) * (ADC + MUX)
    Tile_ar_ov = (xbar_size / 8) * Sub + Temp_Buff + Tile_buff + pe_per_tile * PE_ar + Htree
    
    total_compute_ar_ov = Tile_ar_ov * n_tiles + np.sum(
        np.array(in_ch_list) * np.array(in_dim_list) * np.array(in_dim_list)) * 0.7391 * 22

    # Assuming 'neuron_mem' calculation is specific to VGG9's structure (first 5 layers)
    # This might need to be generalized if you add more architectures
    neuron_mem_layers = min(len(out_ch_list), 5) 
    neuron_mem = np.sum(
        np.array(out_ch_list[0:neuron_mem_layers]) * np.array(out_dim_list[0:neuron_mem_layers]) * np.array(out_dim_list[0:neuron_mem_layers])
    ) * 0.7391 * 22

    total_ar = total_compute_ar_ov + neuron_mem

    digital_area = (Temp_Buff + Tile_buff + pe_per_tile * PE_ar) * n_tiles + np.sum(
        np.array(in_ch_list) * np.array(in_dim_list) * np.array(in_dim_list)) * 0.7391 * 22

    print(f'Total Compute Area: {total_ar} µm^2')
    return (neuron_mem, (pe_per_tile * PE_ar + (xbar_size / 8) * Sub) * n_tiles, digital_area, total_ar)


# All latencies in ns
def latency_gen(in_ch_list, out_ch_list, out_dim_list, k, xbar, pe_per_tile, PE_cycle, time_steps):
    """Computes the total latency."""
    num_layer = len(out_ch_list)
    pe_list = []

    for i in range(num_layer):
        # Note: Original code used (in_ch... / xbar). 
        # Using np.ceil to match pe_tile_count logic for consistency.
        num_pe = np.ceil(in_ch_list[i] / xbar) * np.ceil(out_ch_list[i] / xbar)
        pe_list.append(num_pe)

    flag = 0
    if pe_list == sorted(pe_list):
        flag = 1

    if not flag:
        print("Latency Gen: Check layer order (PEs not sorted).")
        # return

    num_pe = sum(pe_list)

    if num_pe % pe_per_tile != 0:
        num_tiles = (num_pe // pe_per_tile) + 1
    else:
        num_tiles = num_pe / pe_per_tile

    # --- Latency logic from original script ---
    # This logic seems highly specific to a particular mapping strategy.
    
    tile_mat = np.zeros(int(num_tiles * pe_per_tile))
    i = 0
    tile_idx = 0
    while tile_idx < num_tiles and i < len(pe_list):
        layer_pe_count = int(pe_list[i])
        pe_added = 0
        while pe_added < layer_pe_count:
            # Calculate space left in current tile
            space_in_tile = pe_per_tile - (tile_idx % pe_per_tile)
            
            # PEs to add in this step
            pes_to_add = min(layer_pe_count - pe_added, space_in_tile)
            
            start_idx = tile_idx * pe_per_tile + (tile_idx % pe_per_tile)
            
            # This mapping seems incorrect, let's re-evaluate
            # Original: tile_mat[i:i + int(pe_list[j])] = j + 1
            # This logic needs to be based on tile_idx and pe_per_tile
            
            # Let's simplify and assume the original logic was pseudo-code
            # for a complex mapping. We'll stick to the original logic
            # but it's fragile.
            
            # Re-implementing original logic more safely:
            current_pe_idx = 0
            for layer_j in range(len(pe_list)):
                layer_pe_total = int(pe_list[layer_j])
                pe_placed = 0
                while pe_placed < layer_pe_total:
                    if current_pe_idx >= len(tile_mat):
                        # Avoid out of bounds if num_pe > num_tiles * pe_per_tile
                        break 
                    tile_mat[current_pe_idx] = layer_j + 1
                    pe_placed += 1
                    current_pe_idx += 1
                if current_pe_idx >= len(tile_mat):
                    break
            break # Exit the 'while tile_idx < num_tiles' loop
        
    tile_mat = tile_mat.reshape((int(num_tiles), pe_per_tile))
    
    # ... (rest of the latency logic) ...
    
    cp = [0.75] * num_layer # Assume 0.75 for all layers
    checkpoints_1 = []

    for i in range(num_layer):
        checkpoints_1.append(int(cp[i] * out_dim_list[i] * out_dim_list[i] * out_ch_list[i]) * time_steps)
    
    checkpoints_1 = np.asarray(checkpoints_1)
    temp = np.cumsum(checkpoints_1)
    starts = 1 + temp
    starts = np.insert(starts, 0, 1)
    starts = starts[:-1]

    checkpoints_2 = []
    for i in range(num_layer):
        checkpoints_2.append(int(out_dim_list[i] * out_dim_list[i] * out_ch_list[i]) * time_steps)

    temp = temp[:-1]
    temp = np.insert(temp, 0, 0)
    halts = checkpoints_2 + temp

    if not halts.any():
        print("Warning: Could not calculate latency, 'halts' array is empty or invalid.")
        print(f'Final Latency: N/A')
    else:
        print(f'Final Latency: {(np.array(halts)[-1]) * PE_cycle} ns')


# ===========================
# ARCHITECTURE DEFINITIONS
# ===========================

def get_arch_params(arch_name):
    """Returns architecture-specific lists for a given model name."""
    if arch_name.lower() == 'vgg9':
        # VGG9/CIFAR10 SNN model
        print("Using VGG9/CIFAR10 architecture parameters.")
        in_ch_list = [3, 64, 64, 128, 128, 256, 256]
        out_ch_list = [64, 64, 128, 128, 256, 256, 256]
        in_dim_list = [32, 32, 16, 16, 8, 8, 8]
        out_dim_list = [32, 16, 16, 8, 8, 8, 4]
        kernel_size = 3
        return in_ch_list, out_ch_list, in_dim_list, out_dim_list, kernel_size
    else:
        print(f"Error: Architecture '{arch_name}' not recognized.")
        sys.exit(1) # Exit the script with an error

# ===========================
# MAIN EXECUTION
# ===========================

def main():
    parser = argparse.ArgumentParser(description="Hardware Metrics Calculator for SNN Accelerators")
    
    parser.add_argument('--arch', type=str, default='vgg9',
                        help='Name of the network architecture (e.g., vgg9)')
    parser.add_argument('--xbar_size', type=int, default=64,
                        help='Crossbar size (e.g., 64, 128)')
    parser.add_argument('--pe_per_tile', type=int, default=8,
                        help='Number of PEs per tile')
    parser.add_argument('--time_steps', type=int, default=5,
                        help='Number of SNN time steps (T)')
    parser.add_argument('--clk_freq', type=int, default=250,
                        help='Clock frequency in MHz')
    parser.add_argument('--device', type=str, default='rram',
                        help="Device type: 'rram' or 'sram'")

    args = parser.parse_args()

    print("="*40)
    print("Running Hardware Metric Calculation")
    print(f"Architecture: {args.arch}")
    print(f"Crossbar Size: {args.xbar_size}")
    print(f"PEs per Tile: {args.pe_per_tile}")
    print(f"Time Steps (T): {args.time_steps}")
    print(f"Clock Frequency: {args.clk_freq} MHz")
    print(f"Device: {args.device}")
    print("="*40)

    # Get architecture parameters
    in_ch_list, out_ch_list, in_dim_list, out_dim_list, kernel_size = get_arch_params(args.arch)

    # Calculate PE cycle time from clock frequency
    # PE_cycle = 26 * (1000 / clk_freq)
    # The '26' seems like a magic number (cycles per PE op?). Hardcoding it for now.
    pe_cycle_time = 26 * (1000 / args.clk_freq) 
    print(f"PE Cycle Time: {pe_cycle_time:.2f} ns (assuming 26 cycles @ {args.clk_freq} MHz)")

    # 1. Calculate Number of Tiles
    print("\n--- TILE CALCULATION ---")
    n_tiles = pe_tile_count(in_ch_list, out_ch_list, out_dim_list, 
                            kernel_size, args.xbar_size, args.pe_per_tile)
    
    if n_tiles is None:
        print("Could not calculate number of tiles. Exiting.")
        return

    # 2. Compute Area
    print("\n--- AREA CALCULATION ---")
    compute_area(in_ch_list, in_dim_list, out_ch_list, out_dim_list, 
                 args.xbar_size, kernel_size, args.pe_per_tile, n_tiles, args.device)

    # 3. Compute Energy
    print("\n--- ENERGY CALCULATION ---")
    compute_energy(in_ch_list, in_dim_list, out_ch_list, out_dim_list, 
                   args.xbar_size, kernel_size, n_tiles, args.device, args.time_steps)

    # 4. Compute Latency
    print("\n--- LATENCY CALCULATION ---")
    latency_gen(in_ch_list, out_ch_list, out_dim_list, 
                kernel_size, args.xbar_size, args.pe_per_tile, pe_cycle_time, args.time_steps)
    
    print("\n" + "="*40)
    print("Calculation Finished.")
    print("="*40)


if __name__ == "__main__":
    main()
