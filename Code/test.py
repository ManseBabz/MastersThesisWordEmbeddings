import LeaningAlgoImpl.Fast_Text as FT
import LeaningAlgoImpl.Finished_Models as finished


Fast_Text = FT.Fast_Text(dev_mode=True)
Fast_Text.get_model(articles_to_learn=10000, randomTrain=True)
Fast_Text.finished_training()

Fast_Text.clustering_test()


#fin_model = finished.get_model()