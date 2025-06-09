import os
import threading
import subprocess

def main():
    countries = ['any countries you want to use']  # Replace with actual country names
    data_retrieved_path = "file_path_to_retrieved_data"  # Replace with actual path to retrieved data
    data_all_path = "file_path_to_all_data"  # Replace with actual path to all data

    data_retrieved_files = os.listdir(data_retrieved_path)
    data_all_files = os.listdir(data_all_path)
    data_retrieved_rics = set([file[:6] for file in data_retrieved_files])  # Assuming the first 6 characters are the RICs
    data_all_rics = set([file[:6] for file in data_all_files]) # Assuming the first 6 characters are the RICs

    print(len(data_retrieved_rics))
    print(len(data_all_rics))

    # Calculate the difference between all RICs and retrieved RICs
    data_need = data_all_rics - data_retrieved_rics
    print(data_need)
    data_need_list = list(data_need)
    data_need_length = len(data_need_list)
    data_need_remainder = data_need_length % len(countries)

    data_for_each_country = {}
    for i in range(len(countries)):
        data_for_each_country[countries[i]] = []
    
    for sec in data_need_list:
        insert = False
        for i in range(len(countries)):
            # Check if the current country has less data than the average needed
            if len(data_for_each_country[countries[i]]) < data_need_length // len(countries):
                data_for_each_country[countries[i]].append(sec)
                insert = True
                break

        # Separate the remainder into the first few countries
        if not insert:
            data_for_each_country[countries[data_need_remainder-1]].append(sec)
            data_need_remainder -= 1

    for i in range(len(countries)):
        print(f"Country {countries[i]} has {len(data_for_each_country[countries[i]])} data to retrieve")
    
    # Start the threads
    threads = []
    run_crawler_path = "run_crawler.py"  # Replace with the actual path to the run_crawler script
    for i in range(len(countries)):
        target_list = ' '.join(data_for_each_country[countries[i]])
        command = f'start cmd /k "python {run_crawler_path} {countries[i]} {target_list} && pause"'
        t = threading.Thread(target=subprocess.Popen, args=(command, ), kwargs={'shell': True})
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
