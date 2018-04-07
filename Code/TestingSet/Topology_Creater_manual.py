
class topology_creater:
    #Class takes a file of lines of two words with similarity like København Danmark and Stockhold Sverige and creates all possible combinations of those lines,
    #Thereby creating a file to input in a topology test. e.g. Man Woman King Queen
    def read_file_to_array(self, filenamepath):
        file = open(filenamepath, 'r')
        lines = file.read().splitlines()

        return lines

    def generate_topology_combinations(self, lines):
        full_topology = []
        for i in range(0, len(lines)):
            for j in range(0, len(lines)):
                if i != j:
                    full_topology.append(lines[i] + " " + lines[j])

        return full_topology

    def write_array_to_file(self, filenamepath, full_topology, context):
        file = open(filenamepath, "w")
        file.write(context + "\n")
        for line in full_topology:
            file.write(line)
            file.write('\n')
        file.close()



creater = topology_creater()

lines = creater.read_file_to_array('Grammatik6.txt')

full_topology = creater.generate_topology_combinations(lines)

print(full_topology)

creater.write_array_to_file('danish_Grammatik6_topology.txt', full_topology, 'Grammatik6:')
