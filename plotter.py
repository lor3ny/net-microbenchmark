import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob



def DrawLinePlot(data, name):
    print(f"Plotting data collective: {name}")

    # Use a dark theme for the plot
    sns.set_style("whitegrid")  # darker background for axes

    # Create the figure and axes
    f, ax1 = plt.subplots(figsize=(25, 10))
    
    # Convert input data to a DataFrame
    df = pd.DataFrame(data)

    df['cluster_collective'] = df['cluster'].astype(str) + '_' + df['collective'].astype(str)

    # Plot with seaborn
    fig = sns.lineplot(
        data=df,
        x='message_text',
        y='bandwidth',
        hue='cluster_collective',
        style='cluster_collective',
        markers=True,
        markersize=10,
        linewidth=3,
        ax=ax1
    )

    ax1.axhline(
        y=200,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Theoretical Peak {200} Gb/s'
    )

    # Labeling and formatting
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.set_ylabel('Bandwidth (Gb/s)', fontsize=28, labelpad=20)
    ax1.set_xlabel('Message Size', fontsize=28, labelpad=20)
    ax1.set_title(f'{name}', fontsize=38, pad=30)

    # Show legend and layout
    ax1.legend(fontsize=20)
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'plot_lat_{name}.png')  # save with dark background



def LoadData(data, cluster, nodes, path, messages, coll=None):

    print (f"Loading data from {path}")

    for msg in messages:
        msg_mult = msg.strip().split(' ')[1]
        msg_value = msg.strip().split(' ')[0]
        if msg_mult == 'B':
            multiplier = 1
        elif msg_mult == 'KB':
            multiplier = 1024
        elif msg_mult == 'MB':
            multiplier = 1024**2
        elif msg_mult == 'GB':
            multiplier = 1024**3
        else:
            raise ValueError(f"Unknown message size unit in {msg}")

        message_bytes = int(msg_value) * multiplier

        for file_name in os.listdir(path):
            print(f"Processing file: {file_name}")
            file_path = os.path.join(path, file_name)

            found_message_bytes = int(file_name.strip().split("_")[0])
            if found_message_bytes != message_bytes:
                continue

            collective = file_name.strip().split("_")[1].split(".")[0]
            if coll is not None and collective != coll:
                continue

            latencies = []
            with open(file_path, 'r') as file:
                lines = file.readlines()[2:]  # Skip the first 2 lines
                for line in lines:
                    latency = int(line.strip())/1e9
                    latencies.append(latency)
            
            gb_sent = 0
            if collective == 'all2all':
                gb_sent = ((message_bytes / (1024**3))*(nodes-1))*8*1.073741824
            elif collective == 'allgather' or collective == 'reducescatter':
                gb_sent = ((message_bytes / (1024**3))*(nodes-1)/nodes)*8*1.073741824
            elif collective == 'allreduce':
                gb_sent = (2*(message_bytes / (1024**3))*(nodes-1)/nodes)*8*1.073741824

            bandwidth = [gb_sent / x for x in latencies]

            data['latency'].extend(latencies)
            data['bandwidth'].extend(bandwidth)
            data['message_text'].extend([msg]*len(latencies))
            data['message_bytes'].extend([message_bytes]*len(latencies))
            data['cluster'].extend([cluster]*len(latencies))
            data['collective'].extend([collective]*len(latencies))

    return data


def CleanData(data):
    for key in data.keys():
        data[key] = []
    return data

if __name__ == "__main__":

    messages = ['256 B', '2 KB', '16 KB', '128 KB', '512 KB', '1 MB', '8 MB', '16 MB', '64 MB', '128 MB']
    data = {
        'message_text': [],
        'message_bytes': [],
        'latency': [],
        'bandwidth': [],
        'cluster': [],
        'collective': []
    }


    # PLEASE MAKE SURE TO CHANGE THE PATH TO YOUR DATA FOLDER
    
    # If you want to plot all the collectives, uncomment the following lines

    # data_folder = "C:/Users/loren/Desktop/FinalResults/haicgu-ib/2025_04_26__15_34_07/4/"
    # data = LoadData(data, "nanjing", 4 , data_folder, messages)
    # DrawLinePlot(data, "HAICGU-IB 4 Nodes")
    # CleanData(data)


    # #If you want to plot only all2all, uncomment the following lines

    data_folder = "C:/Users/loren/Desktop/nanjing/2025_04_30__15_19_44/4/"
    data = LoadData(data, "nanjing", 4 , data_folder, messages, "all2all")
    DrawLinePlot(data, "Nanjing 4 Nodes")
    CleanData(data)