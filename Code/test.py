import LeaningAlgoImpl.Fast_Text as FT
import LeaningAlgoImpl.Finished_Models as finished
import logging, os


Fast_Text = FT.Fast_Text(dev_mode=True)
Fast_Text.get_model(articles_to_learn=10000, randomTrain=True)
Fast_Text.finished_training()

Fast_Text.clustering_test()


fin_model = finished.Finished_Models()
fin_model.get_model(os.path.dirname(os.path.realpath(__file__))+'/LeaningAlgoImpl/Models/Special_Fast_Text,1,5,1,100,5,None,3,2,6,1,1')
fin_model.clustering_test()