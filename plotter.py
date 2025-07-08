import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import sys



def DrawLinePlot(data, name):
    print(f"Plotting data collective: {name}")

    # Imposta stile e contesto
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Crea figura
    f, ax1 = plt.subplots(figsize=(20, 10))

    # Conversione dati in DataFrame
    df = pd.DataFrame(data)
    df['cluster_collective'] = df['Cluster'].astype(str) + '_' + df['collective'].astype(str)

    # Palette migliorata
    palette = sns.color_palette("Set2", n_colors=df['cluster_collective'].nunique())

    # Lineplot con stile e markers
    sns.lineplot(
        data=df,
        x='message',
        y='bandwidth',
        hue='cluster_collective',
        style='cluster_collective',
        markers=True,
        markersize=10,
        linewidth=3,
        ax=ax1,
        palette=palette
    )

    # Linea teorica
    ax1.axhline(
        y=100,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Theoretical Peak {100} Gb/s'
    )

    # Etichette
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.set_ylabel('Bandwidth (Gb/s)', fontsize=28, labelpad=20)
    ax1.set_xlabel('Message Size', fontsize=28, labelpad=20)
    ax1.set_title(f'{name}', fontsize=38, pad=30)

    # Legenda centrata in basso fuori dal grafico
    ax1.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=3, frameon=True)

    plt.tight_layout()

    # Salvataggio figura
    plt.savefig(f'plots/{name}_line.png', dpi=300, bbox_inches='tight')
    plt.close()



def DrawScatterPlot(data, name):
    print(f"Plotting data collective: {name}")

    # Use a dark theme for the plot
    sns.set_style("whitegrid")  # darker background for axes
    sns.set_context("talk")

    # Create the figure and axes
    f, ax1 = plt.subplots(figsize=(20, 10))
    
    # Convert input data to a DataFrame
    df = pd.DataFrame(data)
    df['cluster_collective'] = df['Cluster'].astype(str) + '_' + df['collective'].astype(str)
    palette = sns.color_palette("Set2", n_colors=df['cluster_collective'].nunique())

    # Plot with seaborn
    fig = sns.scatterplot(
        data=df,
        x='iteration',
        y='bandwidth',
        hue='Cluster',
        style='Message',
        s=80,
        ax=ax1,
        alpha=0.9,
        palette=palette
    )

    # Labeling and formatting
    ax1.tick_params(axis='both', which='major', labelsize=28)
    ax1.set_ylabel('Bandwidth (Gb/s)', fontsize=35, labelpad=20)
    ax1.set_xlabel('Iterations', fontsize=35, labelpad=20)
    ax1.set_title(f'{name}', fontsize=45, pad=30)

    # Show legend and layout
    # Filtra legenda: solo cluster_collective unici + linea teorica

    ax1.legend(
        fontsize=25,           # grandezza testo etichette
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),  # più spazio sotto
        ncol=5,
        frameon=True,
        title=None,
        markerscale=2.0        # ingrandisce i marker nella legenda
    )
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'plots/{name}_scatter.png', dpi=300)  # save with dark background



def LoadData(data, cluster, nodes, path, messages, coll=None, cong=False):

    print (f"Loading data from {path}")

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

            if cong == False and len(file_name.strip().split("_")) == 3:
                continue

            if cong == True and len(file_name.strip().split("_")) == 2:
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
                gb_sent = 2*(message_bytes/1e9)*8

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

    if len(sys.argv) > 1:
        nodes = int(sys.argv[1])
    else:
        nodes = 8


    messages = ['8 B', '64 B', '512 B', '4 KiB', '32 KiB', '256 KiB', '2 MiB', '16 MiB', '128 MiB']
    collectives = ["all2all", "allgather", "reducescatter", "allreduce", "pointpoint"]


    folder_nanjing = f"data/nanjing/4"
    folder_haicgu = f"data/haicgu-eth-1000/4"

    data = LoadData(data, "HAICGU", 4 , folder_haicgu, messages=['128 MiB'], cong=False, coll="allgather")
    data = LoadData(data, "Nanjing", 4 , folder_nanjing, messages=['128 MiB'], cong=False, coll="allgather")
    DrawScatterPlot(data, f"Nanjing vs HAICGU 4 Nodes All-Gather")
    CleanData(data)

    # for coll in collectives:
    #     for mess in messages:  
    #         data = LoadData(data, "HAICGU", nodes , folder_haicgu, [mess], cong=False, coll=coll)
    #         data = LoadData(data, "Nanjing", nodes , folder_nanjing, [mess], cong=False, coll=coll)
    #         DrawScatterPlot(data, f"Nanjing vs HAICGU {nodes} Nodes {coll} {mess}")
    #         CleanData(data)

    # for coll in collectives:
    #     data = LoadData(data, "HAICGU", nodes , folder_haicgu, messages=messages, cong=False, coll=coll)
    #     data = LoadData(data, "Nanjing", nodes , folder_nanjing, messages=messages, cong=False, coll=coll)
    #     DrawLinePlot(data, f"Nanjing vs HAICGU {nodes} Nodes {coll}")
    #     CleanData(data)
