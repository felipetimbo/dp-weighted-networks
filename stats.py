from preproc import statistics

if __name__ == "__main__":
    datasets_names = [
                    'high-school-contacts',
                    # 'copenhagen-interaction',
                     'reality-call2', 
                    # 'contacts-dublin',
                    # 'digg-reply',
                    # 'wiki-talk',
                    # 'sx-stackoverflow',
                    #  'reality-call2' 
                     'enron',
                    # 'wiki-talk',
                     'dblp'
                    ]

    optins_methods = ['affinity']
    optins_perc = [.0]

    statistics = statistics.Statistics(datasets_names, optins_methods, optins_perc)
    statistics.show()