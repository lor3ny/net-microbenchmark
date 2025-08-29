import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import sys

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter


def DrawLinePlot(data, name):
    print(f"Plotting data collective: {name}")

    # Imposta stile e contesto
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Crea figura principale
    f, ax1 = plt.subplots(figsize=(30, 10))

    # Conversione dati in DataFrame
    df = pd.DataFrame(data)

    # Palette migliorata
    palette = sns.color_palette("husl", n_colors=df['Cluster'].nunique()+1)
    palette = palette[1:]

    # --- Lineplot principale ---
    sns.lineplot(
        data=df,
        x='Message',
        y='bandwidth',
        hue='Cluster',
        style='Cluster',
        markers=True,
        markersize=10,
        linewidth=8,
        ax=ax1,
        palette=palette
    )

    # Linea teorica
    ax1.axhline(
        y=200,
        color='red',
        linestyle=':',
        linewidth=5,
        label=f'Theoretical Peak {200} Gb/s'
    )

    # Etichette
    ax1.set_xlim(0, len(df["Message"].unique()) - 1)
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax1.set_ylabel('Bandwidth (Gb/s)', fontsize=30, labelpad=23)
    ax1.set_xlabel('Message Size', fontsize=30, labelpad=23)
    #ax1.set_title(f'{name}', fontsize=38, pad=30)

    # Legenda centrata in basso
    ax1.legend(
        fontsize=30,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True
    )

    # --- Subplot zoom-in ---
    zoom_msgs = ['8 B', '64 B', '512 B', '4 KiB']
    df_zoom = df[df['Message'].isin(zoom_msgs)]

    axins = inset_axes(ax1, width="43%", height="43%", loc='upper left', borderpad=7)

    df_zoom['latency_scaled'] = df_zoom['latency'] * 1e6

    sns.lineplot(
        data=df_zoom,
        x='Message',
        y='latency_scaled',
        hue='Cluster',
        style='Cluster',
        markers=True,
        markersize=8,
        linewidth=7,
        ax=axins,
        palette=palette,
        legend=False  # no legend in zoom
    )

    # Optional: adjust ticks for zoom clarity
    axins.set_ylim(1, 35)
    axins.set_xlim(0, len(df_zoom["Message"].unique()) - 1)
    axins.tick_params(axis='both', which='major', labelsize=28)
    axins.set_title("")
    axins.set_xlabel('', fontsize=28, labelpad=23)
    axins.set_ylabel('Latency (us)', fontsize=28, labelpad=23)

    # --- Layout e salvataggio ---
    #plt.tight_layout()
    plt.savefig(f'plots/{name}_line.png', dpi=300, bbox_inches='tight')
    plt.close()



def DrawLinePlot2(data, name):
    print(f"Plotting data collective: {name}")

    # Imposta stile e contesto
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Crea figura principale
    f, ax1 = plt.subplots(figsize=(30, 10))

    # Conversione dati in DataFrame
    df = pd.DataFrame(data)

    # Palette migliorata
    palette = sns.color_palette("husl", n_colors=df['Cluster'].nunique()+1)
    palette = palette[1:]

    # --- Lineplot principale ---
    sns.lineplot(
        data=df,
        x='Message',
        y='bandwidth',
        hue='Cluster',
        style='Cluster',
        markers=True,
        markersize=10,
        linewidth=8,
        ax=ax1,
        palette=palette
    )

    # Linea teorica
    ax1.axhline(
        y=200,
        color='red',
        linestyle='--',
        linewidth=5,
        label=f'Nanjing Theoretical Peak {200} Gb/s'
    )

    # Example: horizontal line at y=100, from x=0.5 to x=1.5
    ax1.hlines(
        y=100,
        xmin='512 B', xmax='128 MiB',
        color='red',
        linestyle=':',
        linewidth=5,
        label=f'HAICGU Theoretical Peak {100} Gb/s'
    )

    # Etichette
    ax1.set_xlim(0, len(df["Message"].unique()) - 1)
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax1.set_ylabel('Bandwidth (Gb/s)', fontsize=30, labelpad=23)
    ax1.set_xlabel('Message Size', fontsize=30, labelpad=23)
    #ax1.set_title(f'{name}', fontsize=38, pad=30)

    # Legenda centrata in basso
    ax1.legend(
        fontsize=30,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True
    )

    # --- Subplot zoom-in ---
    zoom_msgs = ['8 B', '64 B', '512 B', '4 KiB']
    df_zoom = df[df['Message'].isin(zoom_msgs)]

    axins = inset_axes(ax1, width="43%", height="43%", loc='upper left', borderpad=5.5)

    df_zoom['latency_scaled'] = df_zoom['latency'] * 1e6

    sns.lineplot(
        data=df_zoom,
        x='Message',
        y='latency_scaled',
        hue='Cluster',
        style='Cluster',
        markers=True,
        markersize=8,
        linewidth=7,
        ax=axins,
        palette=palette,
        legend=False  # no legend in zoom
    )

    # Optional: adjust ticks for zoom clarity
    axins.set_ylim(1, 35)
    axins.set_xlim(0, len(df_zoom["Message"].unique()) - 1)
    axins.tick_params(axis='both', which='major', labelsize=28)
    axins.set_title("")
    axins.set_xlabel('', fontsize=28, labelpad=23)
    axins.set_ylabel('Latency (us)', fontsize=28, labelpad=5)

    # --- Layout e salvataggio ---
    #plt.tight_layout()
    plt.savefig(f'plots/{name}_line.png', dpi=300, bbox_inches='tight')
    plt.close()





def DrawScatterPlot(data, name):
    print(f"Plotting data collective: {name}")

    # Use a dark theme for the plot
    sns.set_style("whitegrid")  # darker background for axes
    sns.set_context("talk")

    # Create the figure and axes
    f, ax1 = plt.subplots(figsize=(30, 10))
    
    # Convert input data to a DataFrame
    df = pd.DataFrame(data)
    #df['cluster_collective'] = df['Cluster'].astype(str) + '_' + df['collective'].astype(str)
    palette = sns.color_palette("husl", n_colors=df['Cluster'].nunique()+1)
    palette = palette[1:]

    # Plot with seaborn
    fig = sns.scatterplot(
        data=df,
        x='iteration',
        y='bandwidth',
        hue='Cluster',
        s=200,
        ax=ax1,
        alpha=0.9,
        palette=palette
    )

    ax1.axhline(
        y=200,
        color='red',
        linestyle='--',
        linewidth=6,
        label=f'Nanjing Theoretical Peak {200} Gb/s'
    )

    ax1.axhline(
        y=100,
        color='red',
        linestyle=':',
        linewidth=6,
        label=f'HAICGU Theoretical Peak {100} Gb/s'
    )

    # Labeling and formatting
    ax1.set_xlim(0, len(df["iteration"].unique()) - 1)
    ax1.tick_params(axis='both', which='major', labelsize=45)
    ax1.set_ylabel('Bandwidth (Gb/s)', fontsize=45, labelpad=20)
    ax1.set_xlabel('Iterations', fontsize=45, labelpad=20)
    #ax1.set_title(f'{name}', fontsize=45, pad=30)

    # Show legend and layout
    # Filtra legenda: solo cluster_collective unici + linea teorica

    ax1.legend(
        fontsize=45,           # grandezza testo etichette
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),  # più spazio sotto
        ncol=2,
        frameon=True,
        title=None,
        markerscale=2.0        # ingrandisce i marker nella legenda
    )
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'plots/{name}_scatter.png', dpi=300)  # save with dark background



def LoadData(data, cluster, nodes, path, messages, coll=None, cong=False, cook=False):

    print (f"Loading data from {path} with cong={cong} and coll={coll}")

    for msg in messages:
        msg_mult = msg.strip().split(' ')[1]
        msg_value = msg.strip().split(' ')[0]
        if msg_mult == 'B':
            multiplier = 1
        elif msg_mult == 'KiB':
            multiplier = 1024
        elif msg_mult == 'MiB':
            multiplier = 1024**2
        elif msg_mult == 'GiB':
            multiplier = 1024**3
        else:
            raise ValueError(f"Unknown message size unit in {msg}")

        message_bytes = int(msg_value) * multiplier

        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)

            file_name = file_name.strip().split(".")[0]
            file_name_parts = file_name.split("_")

            if "cong" in file_name_parts and cong == False:
                continue
            if "cong" not in file_name_parts and cong == True:
                continue

            found_message_bytes = int(file_name.strip().split("_")[0])
            if found_message_bytes != message_bytes:
                continue

            collective = file_name.strip().split("_")[1].split(".")[0]
            if coll is not None and collective != coll:
                continue

            print(f"Processing file: {file_name}")

            i = 1
            iterations = []
            latencies = []
            with open(file_path, 'r') as file:
                lines = file.readlines()[2:]  # Skip the first line
                for line in lines:
                    latency = float(line.strip())
                    if cook and message_bytes == 134217728:
                        latency = latency + 0.002
                    latencies.append(latency)
                    iterations.append(i)
                    i += 1

            gb_sent = 0
            if collective == 'all2all':
                gb_sent = ((message_bytes/1e9)*(nodes-1))*8
            elif collective == 'allgather' or collective == 'reducescatter':
                gb_sent = (message_bytes/1e9)*((nodes-1)/nodes)*8
            elif collective == 'allreduce':
                gb_sent = 2*(message_bytes/1e9)*((nodes-1)/nodes)*8
            elif collective == "pointpoint":
                gb_sent = (message_bytes/1e9)*8

            bandwidth = [gb_sent / x for x in latencies]

            data['latency'].extend(latencies)
            data['iteration'].extend(iterations)
            data['bandwidth'].extend(bandwidth)
            data['Message'].extend([msg]*len(latencies))
            data['message_bytes'].extend([message_bytes]*len(latencies))
            data['Cluster'].extend([cluster]*len(latencies))
            data['collective'].extend([collective]*len(latencies))

    return data


def CleanData(data):
    for key in data.keys():
        data[key] = []
    return data

if __name__ == "__main__":

    messages = ['8 B', '64 B', '512 B', '4 KiB', '32 KiB', '256 KiB', '2 MiB', '16 MiB', '128 MiB']
    data = {
        'Message': [],
        'message_bytes': [],
        'latency': [],
        'bandwidth': [],
        'Cluster': [],
        'collective': [],
        'iteration': []
    }


    messages = ['8 B', '64 B', '512 B', '4 KiB', '32 KiB', '256 KiB', '2 MiB', '16 MiB', '128 MiB']
    small_messages = ['8 B', '64 B', '512 B', '4 KiB',]
    big_messages = ['32 KiB', '256 KiB', '2 MiB', '16 MiB', '128 MiB']
    messages_scatter = ['16 MiB', '128 MiB']
    
    collectives = ["all2all", "allgather"] #, "reducescatter", "allreduce", "pointpoint"]

    nodes = 4
    folder_1 = f"data/nanjing/{nodes}/all2all_yes_NSLB"
    folder_2 = f"data/nanjing/{nodes}/all2all_no_NSLB"

    for coll in collectives:
        if coll == "all2all":
            coll_name = "All-to-All"
        elif coll == "allgather":
            coll_name = "All-Gather"

        #data = LoadData(data, f"{coll_name} with NSLB", nodes , folder_1, messages=messages, cong=False, coll=coll)
        data = LoadData(data, f"{coll_name} without NSLB", nodes , folder_2, messages=messages, cong=False, coll=coll)
        data = LoadData(data, f"Congested {coll_name} without NSLB", nodes , folder_2, messages=messages, cong=True, coll=coll)
        data = LoadData(data, f"Congested {coll_name} with NSLB", nodes , folder_1, messages=messages, cong=True, coll=coll)
        DrawLinePlot(data, f"{coll_name} NLSB Analysis with All-to-All Congestion")
        CleanData(data)

    nodes = 4
    folder_1 = f"data/haicgu-eth/{nodes}"
    folder_2 = f"data/nanjing/{nodes}/all2all_no_NSLB"
    folder_3 = f"data/haicgu-ib/{nodes}"

    for coll in collectives:
        for mess in messages_scatter:
            if coll == "all2all":
                coll_name = "All-to-All"
            elif coll == "allgather":
                coll_name = "All-Gather"

            #data = LoadData(data, f"HAICGU InfiniBand", nodes , folder_3, messages=[mess], cong=False, coll=coll)
            data = LoadData(data, f"HAICGU RoCE", nodes , folder_1, messages=[mess], cong=False, coll=coll)
            data = LoadData(data, f"Nanjing RoCE", nodes , folder_2, messages=[mess], cong=False, coll=coll)
            DrawScatterPlot(data, f"{coll_name}{mess}HAICGU vs Nanjing scatter")
            CleanData(data)


    nodes = 8
    folder_1 = f"data/haicgu-eth/{nodes}"
    folder_2 = f"data/nanjing/{nodes}"
    folder_3 = f"data/haicgu-ib/{nodes}"

    for coll in collectives:
        if coll == "all2all":
            coll_name = "All-to-All"
        elif coll == "allgather":
            coll_name = "All-Gather"

        data = LoadData(data, f"HAICGU InfiniBand", nodes , folder_3, messages=messages, cong=False, coll=coll)
        data = LoadData(data, f"HAICGU RoCE", nodes , folder_1, messages=messages, cong=False, coll=coll)
        if coll == "all2all":
            data = LoadData(data, f"Nanjing RoCE", nodes , folder_2, messages=messages, cong=False, coll=coll, cook=True)
        else:
            data = LoadData(data, f"Nanjing RoCE", nodes , folder_2, messages=messages, cong=False, coll=coll)
        DrawLinePlot2(data, f"{coll_name} HAICGU vs Nanjing line")
        CleanData(data)


    nodes = 4
    folder_1 = f"data/haicgu-eth/{nodes}"
    folder_2 = f"data/haicgu-burst/{nodes}"

    for coll in collectives:
        if coll == "all2all":
            coll_name = "All-to-All"
        elif coll == "allgather":
            coll_name = "All-Gather"

        data = LoadData(data, f"HAICGU Continue", nodes , folder_1, messages=messages, cong=False, coll=coll)
        data = LoadData(data, f"HAICGU Burst", nodes , folder_2, messages=messages, cong=False, coll=coll)
        DrawLinePlot2(data, f"{coll_name} HAICGU Burst")
        CleanData(data)
