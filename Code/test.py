import LeaningAlgoImpl.own_fast_text_test as FT


Fast_Text = FT.Fast_Text(dev_mode=False)
Fast_Text.get_model(articles_to_learn=100, randomTrain=True)