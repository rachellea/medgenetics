inputFile = open("kcnh2_reference_sequence_ori.txt", "r") 
exportFile = open("kcnh2_reference_sequence.txt", "w")
seq = ""
counter = 1
for line in inputFile:
    if not line.isupper():
        print(counter)
    seq += line.rstrip()
    counter += 1
exportFile.write(seq) 

inputFile.close()
exportFile.close()
