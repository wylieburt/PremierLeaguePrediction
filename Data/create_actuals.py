import pandas as pd
import numpy as np

def create_all_actuals():
    gw_1_actuals = game_week1()
    gw_2_actuals = game_week2()
    gw_3_actuals = game_week3()
    gw_4_actuals = game_week4()
    gw_5_actuals = game_week5()
    gw_6_actuals = game_week6()
    gw_7_actuals = game_week7()
    gw_8_actuals = game_week8()
    gw_9_actuals = game_week9()
    gw_10_actuals = game_week10()
    gw_11_actuals = game_week11()
    gw_12_actuals = game_week12()
    gw_13_actuals = game_week13()
    gw_14_actuals = game_week14()
    gw_15_actuals = game_week15()
    gw_16_actuals = game_week16()
    gw_17_actuals = game_week17()
    gw_18_actuals = game_week18()
    gw_19_actuals = game_week19()
    gw_20_actuals = game_week20()
    gw_21_actuals = game_week21()
    gw_22_actuals = game_week22()
    
    all_actuals = [gw_1_actuals, 
                   gw_2_actuals, 
                   gw_3_actuals, 
                   gw_4_actuals, 
                   gw_5_actuals, 
                   gw_6_actuals, 
                   gw_7_actuals, 
                   gw_8_actuals,
                   gw_9_actuals,
                   gw_10_actuals,
                   gw_11_actuals,
                   gw_12_actuals,
                   gw_13_actuals,
                   gw_14_actuals,
                   gw_15_actuals,
                   gw_16_actuals,
                   gw_17_actuals,
                   gw_18_actuals,
                   gw_19_actuals,
                   gw_20_actuals,
                   gw_21_actuals,
                   gw_22_actuals]
    
    return all_actuals

def game_week1():   
    # Game week 1
    gw_1_actuals_list = [["Fri 15 Aug","Liverpool", "AFC Bournemouth", "4-2", "Home Win", "Home Win"],
                      ["Sat 16 Aug","Aston Villa", "Newcastle United", "0-0", "Tie", "Home Win"],
                      ["Sat 16 Aug","Brighton & Hove Albion", "Fulham", "1-1", "Tie", "Tie"],
                      ["Sat 16 Aug","Sunderland", "West Ham United", "3-0", "Home Win", "Home Win"],
                      ["Sat 16 Aug","Tottenham Hotspur", "Burnley", "3-0", "Home Win", "Home Win"],
                      ["Sat 16 Aug","Wolverhampton Wanderers", "Manchester City", "0-4", "Away Win", "Away Win"],
                      ["Sat 17 Aug","Nottingham Forest", "Brentford", "3-1", "Home Win", "Tie"], 
                      ["Sat 17 Aug","Chelsea", "Crystal Palace", "0-0", "Tie", "Home Win"],
                      ["Sat 17 Aug","Manchester United", "Arsenal", "0-1", "Away Win", "Away Win"],
                      ["Sat 18 Aug","Leeds United", "Everton", "1-0", "Home Win", "Tie"]]
    gw_1_actuals = pd.DataFrame(gw_1_actuals_list, columns=["Date","Home","Away", "Score", "Result", "Predicted"])
    
    return gw_1_actuals

def game_week2():
    # Game week 2    
    gw_2_actuals_list = [["Fri 22 Aug","West Ham United", "Chelsea", "1-5", "Away Win", "Away Win"],
                      ["Fri 23 Aug","Manchester City", "Tottenham Hotspur", "0-2", "Away Win", "Home Win"],
                      ["Fri 23 Aug","AFC Bournemouth", "Wolverhampton Wanderers", "1-0", "Home Win", "Home Win"],
                      ["Fri 23 Aug","Brentford", "Aston Villa", "1-0", "Home Win", "Away Win"],
                      ["Fri 23 Aug","Burnley", "Sunderland", "2-0", "Home Win", "Home Win"],
                      ["Fri 23 Aug","Arsenal", "Leeds United", "5-0", "Home Win", "Home Win"],
                      ["Fri 24 Aug","Crystal Palace", "Nottingham Forest", "1-1", "Tie", "Tie"], 
                      ["Fri 24 Aug","Everton", "Brighton & Hove Albion", "2-0", "Home Win", "Home Win"],
                      ["Fri 24 Aug","Fulham", "Manchester United", "1-1", "Tie", "Away Win"],
                      ["Fri 25 Aug","Newcastle United", "Liverpool", "2-3", "Away Win", "Away Win"]]
    gw_2_actuals = pd.DataFrame(gw_2_actuals_list, columns=["Date","Home","Away", "Score", "Result", "Predicted"])
    
    return gw_2_actuals

def game_week3():
    # game week 3
    gw_3_actuals_list = [["Sat Aug 30","Chelsea", "Fulham", "2-0", "Home Win", "Home Win"],
                      ["Sat Aug 30", "Manchester United", "Burnley", "3-2", "Home Win", "Home Win"],
                      ["Sat Aug 30", "Sunderland", "Brentford", "2-1", "Home Win", "Home Win"],
                      ["Sat Aug 30", "Tottenham Hotspur", "AFC Bournemouth", "0-1", "Away Win", "Home Win"],
                      ["Sat Aug 30", "Wolverhampton Wanderers", "Everton", "2-3", "Away Win", "Home Win"],
                      ["Sat Aug 30", "Leeds United", "Newcastle United", "0-0", "Tie", "Away Win"],
                      ["Sat Aug 31", "Brighton & Hove Albion", "Manchester City", "2-1", "Home Win", "Away Win"], 
                      ["Sat Aug 31", "Nottingham Forest", "West Ham United", "0-3", "Away Win", "Tie"],
                      ["Sat Aug 31", "Liverpool", "Arsenal", "1-0", "Home Win", "Home Win"],
                      ["Sat Aug 31","Aston Villa", "Crystal Palace", "0-3", "Away Win", "Home Win"]]
    gw_3_actuals = pd.DataFrame(gw_3_actuals_list, columns=["Date","Home","Away", "Score", "Result", "Predicted"])
    
    return gw_3_actuals

def game_week4():
    # game week 4
    gw_4_actuals_list = [["Sat 13 Sep","Arsenal", "Nottingham Forest", "3-0", "Home Win", "Home Win"],
                    ["Sat 13 Sep","AFC Bournemouth", "Brighton & Hove Albion", "2-1", "Home Win", "Away Win"],
                    ["Sat 13 Sep","Crystal Palace", "Sunderland", "0-0", "Tie", "Tie"],
                    ["Sat 13 Sep","Everton", "Aston Villa", "0-0", "Tie", "Away Win"],
                    ["Sat 13 Sep","Fulham", "Leeds United", "1-0", "Home Win", "Home Win"],
                    ["Sat 13 Sep","Newcastle United", "Wolverhampton Wanderers", "1-0", "Home Win", "Home Win"],
                    ["Sat 13 Sep","West Ham United", "Tottenham Hotspur", "0-3", "Away Win", "Away Win"],
                    ["Sat 13 Sep","Brentford", "Chelsea", "2-2", "Tie", "Away Win"],
                    ["Sat 14 Sep","Burnley", "Liverpool", "0-1", "Away Win", "Away Win"],
                    ["Sat 14 Sep","Manchester City", "Manchester United", "3-0", "Home Win", "Home Win"]]
    gw_4_actuals = pd.DataFrame(gw_4_actuals_list, columns=["Date","Home","Away", "Score", "Result", "Predicted"])

    return gw_4_actuals

def game_week5():
    # game week 5
    gw_5_actuals_list = [["Sat 20 Sep 04:30","Liverpool", "Everton", "2-1", "Home Win", "Home Win"],
                      ["Sat 20 Sep 07:00", "Brighton & Hove Albion", "Tottenham Hotspur", "2-2", "Tie", "Away Win"],
                      ["Sat 20 Sep 07:00", "Burnley", "Nottingham Forest", "1-1", "Tie", "Tie"],
                      ["Sat 20 Sep 07:00", "West Ham United", "Crystal Palace", "1-2",  "Away Win", "Away Win"],
                      ["Sat 20 Sep 07:00", "Wolverhampton Wanderers", "Leeds United", "1-3",  "Away Win", "Away Win"],
                      ["Sat 20 Sep 09:30", "Manchester United", "Chelsea", "2-1",  "Home Win", "Away Win"],
                      ["Sat 20 Sep 12:00", "Fulham", "Brentford", "3-1",  "Home Win", "Home Win"], 
                      ["Sun 21 Sep 06:00", "AFC Bournemouth", "Newcastle United", "0-0",  "Tie", "Away Win"],
                      ["Sun 21 Sep 06:00", "Sunderland", "Aston Villa", "1-1",  "Tie", "Away Win"],
                      ["Sun 21 Sep 08:30", "Arsenal", "Manchester City", "1-1",  "Tie", "Away Win"]]
    gw_5_actuals = pd.DataFrame(gw_5_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    return gw_5_actuals

def game_week6():
    # game week 6
    gw_6_actuals_list = [["Sat 27 Sep 04:30","Brentford", "Manchester United", "3-1", "Home Win", "Away Win"],
                      ["Sat 27 Sep 07:00", "Chelsea", "Brighton & Hove Albion",  "1-3", "Away Win", "Home Win"],
                      ["Sat 27 Sep 07:00", "Crystal Palace", "Liverpool", "2-1", "Home Win", "Away Win"],
                      ["Sat 27 Sep 07:00", "Leeds United", "AFC Bournemouth",  "2-2", "Tie", "Tie"],
                      ["Sat 27 Sep 07:00", "Manchester City", "Burnley",  "5-1", "Home Win", "Home Win"],
                      ["Sat 27 Sep 09:30", "Nottingham Forest", "Sunderland",  "0-1", "Away Win", "Tie"],
                      ["Sat 27 Sep 12:00", "Tottenham Hotspur", "Wolverhampton Wanderers",  "1-1", "Tie", "Home Win"], 
                      ["Sun 28 Sep 06:00", "Aston Villa", "Fulham",  "3-1", "Home Win", "Home Win"],
                      ["Sun 28 Sep 08:30", "Newcastle United", "Arsenal",  "1-2", "Away Win", "Away Win"],
                      ["Mon 29 Sep 12:00", "Everton", "West Ham United",  "1-1", "Tie", "Away Win"]]
    gw_6_actuals = pd.DataFrame(gw_6_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])

    return gw_6_actuals

def game_week7():
    # game week 7
    gw_7_actuals_list = [["Fri 03 Oct 04:30","AFC Bournemouth", "Fulham", "3-1", "Home Win", "Tie"],
                      ["Sat 04 Oct 12:00", "Leeds United", "Tottenham Hotspur",  "1-2", "Away Win", "Away Win"],
                      ["Sat 04 Oct 07:00", "Arsenal", "West Ham United", "2-0", "Home Win", "Home Win"],
                      ["Sat 04 Oct 07:00", "Manchester United", "Sunderland",  "2-0", "Home Win", "Home Win"],
                      ["Sat 04 Oct 09:30", "Chelsea", "Liverpool",  "2-1", "Home Win", "Away Win"],
                      ["Sun 05 Oct 06:00", "Aston Villa", "Burnley",  "2-1", "Home Win", "Home Win"],
                      ["Sun 05 Oct 06:00", "Everton", "Crystal Palace",  "2-1", "Home Win", "Tie"], 
                      ["Sun 05 Oct 06:00", "Newcastle United", "Nottingham Forest",  "2-0", "Home Win", "Home Win"],
                      ["Sun 05 Oct 06:00", "Wolverhampton Wanderers", "Brighton & Hove Albion",  "1-1", "Tie", "Away Win"],
                      ["Sun 05 Oct 08:30", "Brentford", "Manchester City",  "0-1", "Away Win", "Away Win"]]
    gw_7_actuals = pd.DataFrame(gw_7_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])

    return gw_7_actuals

def game_week8():
    # game week 8
    gw_8_actuals_list = [["Sat 18 Oct 04:30","Nott'ham Nottingham Forest", "Chelsea", "0-3", "Away Win", "Away Win"],
                      ["Sat 18 Oct 07:00", "Brighton & Hove Albion", "Newcastle United",  "2-1", "Home Win", "Away Win"],
                      ["Sat 18 Oct 07:00", "Burnley", "Leeds United", "2-0", "Home Win", "Tie"],
                      ["Sat 18 Oct 07:00", "Crystal Palace", "AFC Bournemouth",  "3-3", "Tie", "Tie"],
                      ["Sat 18 Oct 07:00", "Manchester City", "Everton",  "2-0", "Home Win", "Home Win"],
                      ["Sat 18 Oct 07:00", "Sunderland", "Wolverhampton Wanderers",  "2-0", "Home Win", "Tie"],
                      ["Sat 18 Oct 09:30", "Fulham", "Arsenal",  "0-1", "Away Win", "Away Win"], 
                      ["Sun 19 Oct 06:00", "Spurs", "Aston Villa",  "1-2", "Away Win", "Home Win"],
                      ["Sun 19 Oct 08:30", "Liverpool", "Manchester United",  "1-2", "Away Win", "Home Win"],
                      ["Mon 20 Oct 12:00", "West Ham United", "Brentoford",  "0-2", "Away Win", "Tie"]]
    gw_8_actuals = pd.DataFrame(gw_8_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    return gw_8_actuals

def game_week9():
    # game week 8
    gw_9_actuals_list = [["Fri 24 Oct 12:00","Leeds United", "West Ham United", "2-1", "Home Win", "Tie"],
                      ["Sat 25 Oct 07:00", "Chelsea", "Sunderland",  "1-2", "Away Win", "Home Win"],
                      ["Sat 25 Oct 07:00", "Newcastle United", "Fulham", "2-1", "Home Win", "Home Win"],
                      ["Sat 25 Oct 09:30", "Manchester United", "Brighton & Hove Albion", "4-2", "Home Win", "Home Win"],
                      ["Sat 25 Oct 12:00", "Brentford", "Liverpool",  "3-2", "Home Win", "Away Win"],
                      ["Sun 26 Oct 07:00", "AFC Bournemouth", "Nottingham Forest",  "2-0", "Home Win", "Tie"],
                      ["Sun 26 Oct 07:00", "Arsenal", "Crystal Palace",  "1-0", "Home Win", "Home Win"], 
                      ["Sun 26 Oct 07:00", "Aston Villa", "Manchester City",  "1-0", "Home Win", "Away Win"],
                      ["Sun 26 Oct 07:00", "Wolverhampton Wanderers", "Burnley",  "2-3", "Away Win", "Tie"],
                      ["Sun 26 Oct 09:30", "Everton", "Tottenham Hotspur",  "0-3", "Away Win", "Away Win"]]
    gw_9_actuals = pd.DataFrame(gw_9_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    return gw_9_actuals

def game_week10():
    # game week 8
    gw_10_actuals_list = [["Sat 1 Nov 08:00","Brighton & Hove Albion", "Leeds United", "3-0", "Home Win", "Tie"],
                      ["Sat 1 Nov 08:00", "Burnley", "Arsenal",  "0-2", "Away Win", "Away Win"],
                      ["Sat 1 Nov 08:00", "Crystal Palace", "Brentford", "2-0", "Home Win", "Tie"],
                      ["Sat 1 Nov 08:00", "Fulham", "Wolverhampton Wanderers",  "3-0", "Home Win", "Tie"],
                      ["Sat 1 Nov 08:00", "Nottingham Forest", "Manchester United",  "2-2", "Tie", "Away Win"],
                      ["Sat 1 Nov 10:30", "Tottenham Hotspur", "Chelsea",  "0-1", "Away Win", "Away Win"],
                      ["Sat 1 Nov 13:00", "Liverpool", "Aston Villa",  "2-0", "Home Win", "Home Win"], 
                      ["Sun 2 Nov 06:00", "West Ham United", "Newcastle United",  "3-1", "Home Win", "Away Win"],
                      ["Sun 2 Nov 08:30", "Manchester City", "AFC Bournemouth",  "3-1", "Home Win", "Home Win"],
                      ["Mon 3 Nov 12:00", "Sunderland", "Everton",  "1-1", "Tie", "Tie"]]
    gw_10_actuals = pd.DataFrame(gw_10_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    return gw_10_actuals

def game_week11():
    # game week 8
    gw_11_actuals_list = [["Sat 8 Nov","Spurs", "Man United", "2-2", "Tie", "Away Win"],
                      ["Sat 8 Nov", "Everton", "Fulham",  "2-0", "Home Win", "Tie"],
                      ["Sat 8 Nov", "West Ham", "Burnley", "3-2", "Home Win", "Tie"],
                      ["Sat 8 Nov", "Sunderland", "Arsenal",  "2-2", "TIe", "Away Win"],
                      ["Sat 8 Nov", "Chelsea", "Wolves",  "3-0", "Home Win", "Home Win"],
                      ["Sun 9 Nov", "Aston Villa", "Bournemouth",  "4-0", "Home Win", "Home Win"],
                      ["Sun 9 Nov", "Brentford", "Newcastle",  "3-1", "Home Win", "Home Win"], 
                      ["Sun 9 Nov", "Crystal Palace", "Brighton",  "0-0", "Tie", "Tie"],
                      ["Sun 9 Nov", "Nott'm Forest", "Leeds",  "3-1", "Home Win", "Tie"],
                      ["Sun 9 Nov", "Man City", "Liverpool",  "3-0", "Home Win", "Home Win"]]
    gw_11_actuals = pd.DataFrame(gw_11_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    return gw_11_actuals

def game_week12():
    # game week 8
    gw_12_actuals_list = [["Sat 22 Nov", "Burnley", "Chelsea",  "0-2", "Away Win", "Away Win"],
                      ["Sat 22 Nov", "Bournemouth", "West Ham",  "2-2", "Tie", "Home Win"],
                      ["Sat 22 Nov", "Brighton", "Brentford",  "2-1", "Home Win", "Away Win"],
                      ["Sat 22 Nov", "Fulham", "Sunderland",  "1-0", "Home Win", "Tie"],
                      ["Sat 22 Nov", "Liverpool", "Nott'm Forest",  "0-3", "Away Win", "Home Win"],
                      ["Sat 22 Nov", "Wolves", "Crystal Palace",  "0-2", "Away Win", "Tie"],
                      ["Sat 22 Nov", "Newcastle", "Man City",  "2-1", "Home Win", "Away Win"], 
                      ["Sun 22 Nov", "Leeds", "Aston Villa",  "1-2", "Away Win", "Away Win"],
                      ["Sun 23 Nov", "Arsenal", "Spurs",  "4-1", "Home Win", "Home Win"],
                      ["Mon 24 Nov", "Man United", "Everton",  "0-1", "Away Win", "Home Win"]]
    gw_12_actuals = pd.DataFrame(gw_12_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    return gw_12_actuals    

def game_week13():
    # game week 8
    gw_13_actuals_list = [["Sat 29 Nov", "Brentford", "Burnley",  "3-1", "Home Win", "Home Win"],
                      ["Sat 29 Nov", "Man City", "Leeds",  "3-2", "Home Win", "Home Win"],
                      ["Sat 29 Nov", "Sunderland", "Bournmouth",  "3-2", "Home Win", "Away Win"],
                      ["Sat 29 Nov", "Everton", "Newcastle",  "1-4", "Away Win", "TIe"],
                      ["Sat 29 Nov", "Spurs", "Fulham",  "1-2", "Away Win", "Tie"],
                      ["Sun 30 Nov", "Crystal Palace", "Man United",  "1-2", "Away Win", "Away Win"],
                      ["Sun 30 Nov", "Aston Villa", "Wolves",  "1-0", "Home Win", "Home Win"], 
                      ["Sun 30 Nov", "Nott'm Forest", "Brighton",  "0-2", "Away Win", "Tie"],
                      ["Sun 30 Nov", "West Ham", "Liverpool",  "0-2", "Away Win", "Away Win"],
                      ["Sun 30 Nov", "Chelsea", "Arsenal",  "1-1", "Tie", "Home Win"]]
    gw_13_actuals = pd.DataFrame(gw_13_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    return gw_13_actuals   

def game_week14():
    # game week 8
    gw_14_actuals_list = [["Tue 2 Dec", "Bournemouth", "Everton",  "0-1", "Away Win", "Home Win"],
                      ["Tue 2 Dec", "Fulham", "Man City",  "4-5", "Away Win", "Away Win"],
                      ["Tue 2 Dec", "Newcastle", "Spurs",  "2-2", "Tie", "Home Win"],
                      ["Wed 3 Dec", "Arsenal", "Brentford",  "2-0", "Home Win", "Away Win"],
                      ["Wed 3 Dec", "Brighton", "Aston Villa",  "3-4", "Away Win", "Away Win"],
                      ["Wed 3 Dec", "Burnley", "Crystal Palace",  "0-1", "Away Win", "Away Win"],
                      ["Wed 3 Dec", "Wolves", "Nott'm Forest",  "0-1", "Away Win", "Tie"], 
                      ["Wed 3 Dec", "Leeds", "Chelsea",  "3-1", "Home Win", "Away Win"],
                      ["Wed 3 Dec", "Liverpool", "Sunderland",  "1-1", "Tie", "Home Win"],
                      ["Thu 4 Dec", "Man United", "West Ham",  "1-1", "Tie", "Home Win"]]
    gw_14_actuals = pd.DataFrame(gw_14_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    return gw_14_actuals   

def game_week15():
    # game week 8
    gw_15_actuals_list = [["Sat 6 Dec", "Aston Villa", "Arsenal",  "2-1", "Home Win", "Home Win"],
                      ["Sat 6 Dec", "Bournemouth", "Chelsea",  "0-0", "Tie", "Home WIn"],
                      ["Sat 6 Dec", "Everton", "Nott'm Forest",  "3-0", "Home Win", "Tie"],
                      ["Sat 6 Dec", "Man City", "Sunderland",  "3-0", "Home Win", "Home Win"],
                      ["Sat 6 Dec", "Newcastle", "Burnley",  "2-1", "Home Win", "Home Win"],
                      ["Sat 6 Dec", "Spurs", "Brentford",  "2-0", "Home Win", "Away Win"],
                      ["Sat 6 Dec", "Leeds", "Liverpool",  "3-3", "Tie", "Away Win"], 
                      ["Sun 7 Dec", "Brighton", "West Ham",  "1-1", "Tie", "Tie"],
                      ["Sun 7 Dec", "Fulham", "Crystal Palace",  "1-2", "Away Win", "Tie"],
                      ["Mon 8 Dec", "Wolves", "Man United",  "1-4", "Away Win", "Away Win"]]
    gw_15_actuals = pd.DataFrame(gw_15_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    return gw_15_actuals   

def game_week16():
    # game week 8
    gw_16_actuals_list = [["Thu 4 Dec", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"], 
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"]]
    gw_16_actuals = pd.DataFrame(gw_16_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    gw_16_actuals = pd.read_csv('Data/gw_16_actuals.csv')
    
    return gw_16_actuals   

def game_week17():
    # game week 8
    gw_17_actuals_list = [["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"], 
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"]]
    gw_17_actuals = pd.DataFrame(gw_17_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    gw_17_actuals = pd.read_csv('Data/gw_17_actuals.csv')

    return gw_17_actuals   

def game_week18():
    # game week 8
    gw_18_actuals_list = [["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"], 
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"]]
    gw_18_actuals = pd.DataFrame(gw_18_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    gw_18_actuals = pd.read_csv('Data/gw_18_actuals.csv')

    return gw_18_actuals   

def game_week19():
    # game week 8
    gw_19_actuals_list = [["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"], 
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"]]
    gw_19_actuals = pd.DataFrame(gw_19_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    gw_19_actuals = pd.read_csv('Data/gw_19_actuals.csv')

    return gw_19_actuals   

def game_week20():
    # game week 8
    gw_20_actuals_list = [["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"], 
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"]]
    gw_20_actuals = pd.DataFrame(gw_20_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    gw_20_actuals = pd.read_csv('Data/gw_20_actuals.csv')

    return gw_20_actuals   

def game_week21():
    # game week 8
    gw_21_actuals_list = [["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"], 
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"]]
    gw_21_actuals = pd.DataFrame(gw_21_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    gw_21_actuals = pd.read_csv('Data/gw_21_actuals.csv')

    return gw_21_actuals   

def game_week22():
    # game week 8
    gw_22_actuals_list = [["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sat 8 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"], 
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"],
                      ["Sun 9 Nov", "", "",  "", "", "--"]]
    gw_22_actuals = pd.DataFrame(gw_22_actuals_list, columns=["Date", "Home","Away", "Score", "Result", "Predicted"])
    
    return gw_22_actuals  