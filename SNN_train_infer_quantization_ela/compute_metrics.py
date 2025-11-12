import numpy as np
import argparse
import ast  # <-- Make sure this import is here

# ... (All your functions: pe_tile_count, compute_energy, compute_area, latency_gen) ...
# (No changes to the functions themselves)

def pe_tile_count(in_ch_list, out_ch_list, out_dim_list, k, xbar, pe_per_tile):
    num_layer = len(out_ch_list)
    pe_list = []
    for i in range(num_layer):
        num_pe = np.ceil(in_ch_list[i] / xbar) * np.ceil(out_ch_list[i] / xbar)
        pe_list.append(num_pe)
    flag = 0
    if (pe_list == sorted(pe_list)):
        flag = 1
    if (flag):
        print("Layers in order")
    else:
        print("Check layer order")
        return
    num_pe = sum(pe_list)
    print(f'No. of PEs {num_pe}')
    if (num_pe % pe_per_tile != 0):
        num_tiles = (num_pe // pe_per_tile) + 1
    else:
        num_tiles = num_pe / pe_per_tile
    print(f'No. of Tiles {num_tiles}')
    return num_tiles

def compute_energy(in_ch_list, in_dim_list, out_ch_list, out_dim_list, xbar_size, k, n_tiles, device, time_steps):
    if device == 'rram':
        xbar_ar = 1.76423
    elif device == 'sram':
        xbar_ar = 671.089
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
    for i in range(len(out_ch_list)):
        Total_PE_cycle = np.ceil(out_ch_list[i] / xbar_size) * np.ceil(in_ch_list[i] / xbar_size) * (
                out_dim_list[i] * out_dim_list[i])
        tot_energy += Total_PE_cycle * PE_cycle_energy * time_steps
        tot_pe_cycle += Total_PE_cycle
        energy_layerwise.append(Total_PE_cycle * PE_cycle_energy * time_steps)
    print(f'total_energy {tot_energy} pJ')
    return energy_layerwise

def compute_area(in_ch_list, in_dim_list, out_ch_list, out_dim_list, xbar_size, k, pe_per_tile, n_tiles, device):
    if device == 'rram':
        xbar_ar = 26.2144
    elif device == 'sram':
        xbar_ar = 671.089
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
    neuron_mem = np.sum(
        np.array(out_ch_list[0:5]) * np.array(out_dim_list[0:5]) * np.array(out_dim_list[0:5])) * 0.7391 * 22
    total_ar = total_compute_ar_ov + neuron_mem
    digital_area = (Temp_Buff + Tile_buff + pe_per_tile * PE_ar) * n_tiles + np.sum(
        np.array(in_ch_list) * np.array(in_dim_list) * np.array(in_dim_list)) * 0.7391 * 22
    print(f'total_compute_area {total_ar} µm^2')
    return (neuron_mem, (pe_per_tile * PE_ar + (xbar_size / 8) * Sub) * n_tiles, digital_area, total_ar)

def latency_gen(in_ch_list, out_ch_list, out_dim_list, k, xbar, pe_per_tile, PE_cycle, time_steps):
    num_layer = len(out_ch_list)
    pe_list = []
    for i in range(num_layer):
        num_pe = (in_ch_list[i] / xbar) * (out_ch_list[i] / xbar)
        pe_list.append(num_pe)
    flag = 0
    if (pe_list == sorted(pe_list)):
        flag = 1
    if (flag):
        print("Layers in order")
    else:
        print("Check layer order")
        return
    num_pe = sum(pe_list)
    if (num_pe % pe_per_tile != 0):
        num_tiles = (num_pe // pe_per_tile) + 1
    else:
        num_tiles = num_pe / pe_per_tile
    tile_mat = np.zeros(int(num_tiles * pe_per_tile))
    i = 0
    while (i < num_tiles):
        for j in range(len(pe_list)):
            tile_mat[i:i + int(pe_list[j])] = j + 1
            i = i + int(pe_list[j])
    tile_mat = tile_mat.reshape((int(num_tiles), pe_per_tile))
    tile_dist = np.zeros((int(num_tiles), num_layer))
    for i in range(int(num_tiles)):
        for j in range(num_layer):
            tile_dist[i, j] = np.sum(tile_mat[i] == (j + 1))
    mult_tile = []
    for i in range(num_layer):
        if (np.sum(tile_dist[:, i] != 0) == 1):
            mult_tile.append(0)
        else:
            mult_tile.append(1)
    cp = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
    active_layer = np.zeros(num_layer)
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
    ckpt = []
    for i in range(len(halts)):
        ckpt.append(starts[i])
        ckpt.append(halts[i])
    ckpt = np.sort(ckpt)
    print(f' Final Latency {(np.array(halts)[-1]) * PE_cycle} ns')


def main(args):
    """
    Main function to run the computations based on provided arguments.
    """
    
    # <-- THIS IS THE CRITICAL CHANGE
    try:
        in_ch_list = ast.literal_eval(args.in_ch_list)
        out_ch_list = ast.literal_eval(args.out_ch_list)
        in_dim_list = ast.literal_eval(args.in_dim_list)
        out_dim_list = ast.literal_eval(args.out_dim_list)
    except Exception as e:
        print(f"Error parsing list arguments. Please check formatting. Ensure they are quoted strings '[]'.")
        print(f"Error: {e}")
        return
        
    # --- Retrieve other parameters ---
    xbar_size = args.xbar_size
    kernel_size = args.kernel_size
    pe_per_tile = args.pe_per_tile
    time_steps = args.time_steps
    clk_freq = args.clk_freq
    device = args.device

    # Simple validation
    if not (len(in_ch_list) == len(out_ch_list) == len(in_dim_list) == len(out_dim_list)):
        print("Error: All layer list arguments (in_ch_list, out_ch_list, in_dim_list, out_dim_list) must have the same number of elements.")
        return

    # Calculate PE_cycle based on clk_freq
    PE_cycle = 26 * (1000 / clk_freq)

    print("--- Calculating Tile Count ---")
    n_tiles = pe_tile_count(in_ch_list, out_ch_list, out_dim_list, kernel_size, xbar_size, pe_per_tile)
    
    if n_tiles is None:
        print("Exiting due to layer order issue.")
        return
    
    print("\n--- Calculating Area ---")
    compute_area(in_ch_list, in_dim_list, out_ch_list, out_dim_list, xbar_size, kernel_size, pe_per_tile, n_tiles, device)
    
    print("\n--- Calculating Energy ---")
    compute_energy(in_ch_list, in_dim_list, out_ch_list, out_dim_list, xbar_size, kernel_size, n_tiles, device, time_steps)
    
    print("\n--- Calculating Latency ---")
    latency_gen(in_ch_list, out_ch_list, out_dim_list, kernel_size, xbar_size, pe_per_tile, PE_cycle, time_steps)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Calculate Energy, Area, and Latency for a Neuromorphic Accelerator Model.")

    # <-- THIS IS THE OTHER CRITICAL CHANGE
    # Note 'type=str' and no 'nargs=+'
    parser.add_argument('--in_ch_list', type=str, default='[3, 64, 64, 128, 128, 256, 256]',
                        help='List of input channels for each layer, passed as a string.')
    parser.add_argument('--out_ch_list', type=str, default='[64, 64, 128, 128, 256, 256, 256]',
                        help='List of output channels for each layer, passed as a string.')
    parser.add_argument('--in_dim_list', type=str, default='[32, 32, 16, 16, 8, 8, 8]',
                        help='List of input dimensions for each layer, passed as a string.')
    parser.add_argument('--out_dim_list', type=str, default='[32, 16, 16, 8, 8, 8, 4]',
                        help='List of output dimensions for each layer, passed as a string.')
    
    # --- Other Arguments (unchanged) ---
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Kernel size (e.g., 3x3 convolution).')
    parser.add_argument('--xbar_size', type=int, default=64,
                        help='Size of the crossbar array (e.g., 64x64).')
    parser.add_argument('--pe_per_tile', type=int, default=8,
                        help='Number of Processing Elements (PEs) per tile.')
    parser.add_argument('--device', type=str, default='rram', choices=['rram', 'sram'],
                        help='The memory device type (rram or sram) for area/energy models.')
    parser.add_argument('--time_steps', type=int, default=5,
                        help='Number of time steps for SNN simulation.')
    parser.add_argument('--clk_freq', type=float, default=250,
                        help='Clock frequency in MHz.')

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function
    main(args)
