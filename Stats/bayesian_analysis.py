import numpy as np
import pandas as pd
from scipy import stats
import joblib

class BayesianTeamAnalyzer:
    """
    Bayesian analysis for comparing teams with different sample sizes.
    Handles the problem where teams with fewer matches appear artificially superior.
    """
    
    def __init__(self, stat_names=None):
        self.stat_names = stat_names or ['SoT', 'GF', 'Poss', 'Long_Completes', 'Success', 'Blocks']
        self.results = {}
    
    def empirical_bayes_shrinkage(self, team_data, prior_mean, prior_precision=1.0):
        """
        Apply empirical Bayes shrinkage to team statistics.
        
        Parameters:
        - team_data: array of match-level statistics for the team
        - prior_mean: league average (or dominant team's average as prior)
        - prior_precision: confidence in the prior (higher = more shrinkage)
        """
        n = len(team_data)
        sample_mean = np.mean(team_data)
        sample_var = np.var(team_data, ddof=1) if n > 1 else 1.0
        
        # Empirical Bayes shrinkage formula
        # Shrinkage intensity depends on sample size and variance
        tau = prior_precision  # Prior precision
        likelihood_precision = n / sample_var if sample_var > 0 else n
        
        # Posterior mean (shrunk estimate)
        posterior_precision = tau + likelihood_precision
        shrunk_mean = (tau * prior_mean + likelihood_precision * sample_mean) / posterior_precision
        
        # Calculate shrinkage factor for interpretation
        shrinkage_factor = tau / posterior_precision
        
        return {
            'raw_mean': sample_mean,
            'shrunk_mean': shrunk_mean,
            'shrinkage_factor': shrinkage_factor,
            'posterior_precision': posterior_precision,
            'n_matches': n
        }
    
    def hierarchical_bayes_estimate(self, all_teams_data, target_team_data):
        """
        Hierarchical Bayesian model using all teams to inform priors.
        
        Parameters:
        - all_teams_data: list of arrays, each containing one team's data
        - target_team_data: array of target team's data to estimate
        """
        # Estimate hyperpriors from all teams
        all_means = [np.mean(team) for team in all_teams_data if len(team) > 0]
        
        # Population-level parameters
        mu_0 = np.mean(all_means)  # Population mean
        tau_0 = 1 / np.var(all_means) if len(all_means) > 1 else 1.0  # Population precision
        
        # Apply shrinkage using population parameters as prior
        return self.empirical_bayes_shrinkage(target_team_data, mu_0, tau_0)
    
    def compare_teams_bayesian(self, team1_data, team2_data, stat_names=None):
        """
        Compare two teams using Bayesian analysis.
        Handles different sample sizes appropriately.
        """
        if stat_names is None:
            stat_names = self.stat_names
            
        results = {}
        
        for i, stat in enumerate(stat_names):
            # Extract statistic for both teams
            team1_stat = [match[i] for match in team1_data]
            team2_stat = [match[i] for match in team2_data]
            
            # Use Team 1 as prior (since it has more data)
            team1_mean = np.mean(team1_stat)
            
            # Apply Bayesian shrinkage to both teams
            team1_result = self.empirical_bayes_shrinkage(team1_stat, team1_mean, prior_precision=0.1)
            team2_result = self.empirical_bayes_shrinkage(team2_stat, team1_mean, prior_precision=1.0)
            
            results[stat] = {
                'team1': team1_result,
                'team2': team2_result,
                'difference_raw': team2_result['raw_mean'] - team1_result['raw_mean'],
                'difference_shrunk': team2_result['shrunk_mean'] - team1_result['shrunk_mean']
            }
        
        self.results = results
        return results
    
    def credible_intervals(self, team_data, prior_mean, prior_precision=1.0, confidence=0.95):
        """
        Calculate Bayesian credible intervals (analogous to confidence intervals).
        """
        n = len(team_data)
        sample_mean = np.mean(team_data)
        sample_var = np.var(team_data, ddof=1) if n > 1 else 1.0
        
        # Posterior parameters
        tau = prior_precision
        likelihood_precision = n / sample_var if sample_var > 0 else n
        posterior_precision = tau + likelihood_precision
        posterior_mean = (tau * prior_mean + likelihood_precision * sample_mean) / posterior_precision
        posterior_var = 1 / posterior_precision
        
        # Credible interval
        alpha = 1 - confidence
        z = stats.norm.ppf(1 - alpha/2)
        margin = z * np.sqrt(posterior_var)
        
        return {
            'mean': posterior_mean,
            'lower': posterior_mean - margin,
            'upper': posterior_mean + margin,
            'margin': margin
        }

    
    def summary_table(self):
        """
        Create a summary table of the Bayesian analysis.
        """
        if not self.results:
            print("No results available. Run compare_teams_bayesian first.")
            return None
        
        summary_data = []
        team1_data_list = []
        team2_data_list = []
        
        for stat, data in self.results.items():
            summary_data.append
        
        for stat, data in self.results.items():
            summary_data.append({
                'Statistic': stat,
                'Team1_Raw': f"{data['team1']['raw_mean']:.2f}",
                'Team1_Bayesian': f"{data['team1']['shrunk_mean']:.2f}",
                'Team2_Raw': f"{data['team2']['raw_mean']:.2f}",
                'Team2_Bayesian': f"{data['team2']['shrunk_mean']:.2f}",
                'Raw_Difference': f"{data['difference_raw']:.2f}",
                'Bayesian_Difference': f"{data['difference_shrunk']:.2f}",
                'Team2_Shrinkage': f"{data['team2']['shrinkage_factor']:.2f}"
            })
            team1_data_list.append(data['team1']['shrunk_mean'])
            team2_data_list.append(data['team2']['shrunk_mean'])
        
        df = pd.DataFrame(summary_data)
        time1_baye_df = pd.DataFrame([team1_data_list], columns=["SoT", "GF", "Poss", "Long_Cmp", "Succ", "Blocks"])
        time2_baye_df = pd.DataFrame([team2_data_list], columns=["SoT", "GF", "Poss", "Long_Cmp", "Succ", "Blocks"])

        return df, time1_baye_df, time2_baye_df

