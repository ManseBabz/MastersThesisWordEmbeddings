import LeaningAlgoImpl.Finished_Models as FM
import logging, os, time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

finished_model = FM.Finished_Models()
finished_model.get_model(dir_path + '/Code/Models/CBOW,0,5,0,10,100,1,10000_Trained_on1000000articles')

print(finished_model)

finished_model.acc()#'danish-topology.txt')

print(finished_model.acc('questions-words1.txt'))

#print(finished_model.acc('danish-topology.txt'))

#finished_model.get_vocabulary()

print('similarity')
print(finished_model.similarity('en', 'er'))

print('english trial')
print(finished_model.human_similarity_test())

time.sleep(10)
print('danish trial')
print(finished_model.human_similarity_test('danish-similarity-test.tsv'))
