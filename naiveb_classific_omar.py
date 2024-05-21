class BayesCalculator:
    def __init__(self):
        self.parameters = {}

    def set_parameter(self, key, value):
        self.parameters[key] = value

    def calculate_all_probabilities(self):
        results = {}

        # Extract known parameters
        P_A = self.parameters.get('P(A)', None)
        P_B = self.parameters.get('P(B)', None)
        P_C = self.parameters.get('P(C)', None)
        P_D = self.parameters.get('P(D)', None)
        P_F = self.parameters.get('P(F)', None)
        P_not_A = self.parameters.get('P(¬A)', 1 - P_A if P_A is not None else None)
        P_not_B = self.parameters.get('P(¬B)', 1 - P_B if P_B is not None else None)
        P_not_C = self.parameters.get('P(¬C)', 1 - P_C if P_C is not None else None)
        P_not_D = self.parameters.get('P(¬D)', 1 - P_D if P_D is not None else None)
        P_not_F = self.parameters.get('P(¬F)', 1 - P_F if P_F is not None else None)
        P_A_given_B = self.parameters.get('P(A|B)', None)
        P_A_given_not_B = self.parameters.get('P(A|¬B)', None)
        P_B_given_A = self.parameters.get('P(B|A)', None)
        P_B_given_not_A = self.parameters.get('P(B|¬A)', None)
        P_C_given_A = self.parameters.get('P(C|A)', None)
        P_A_given_C = self.parameters.get('P(A|C)', None)
        P_D_given_A = self.parameters.get('P(D|A)', None)
        P_A_given_D = self.parameters.get('P(A|D)', None)
        P_F_given_A = self.parameters.get('P(F|A)', None)
        P_A_given_F = self.parameters.get('P(A|F)', None)
        P_C_given_B = self.parameters.get('P(C|B)', None)
        P_B_given_C = self.parameters.get('P(B|C)', None)
        P_D_given_C = self.parameters.get('P(D|C)', None)
        P_D_given_B = self.parameters.get('P(D|B)', None)
        P_D_given_F = self.parameters.get('P(D|F)', None)
        P_C_given_D = self.parameters.get('P(C|D)', None)

        # Calculate P(¬A), P(¬B), P(¬C), P(¬D), P(¬F)
        if P_A is not None:
            results['P(A)'] = P_A
            P_not_A = 1 - P_A
            results['P(¬A)'] = P_not_A

        if P_B is not None:
            results['P(B)'] = P_B
            P_not_B = 1 - P_B
            results['P(¬B)'] = P_not_B

        if P_C is not None:
            results['P(C)'] = P_C
            P_not_C = 1 - P_C
            results['P(¬C)'] = P_not_C

        if P_D is not None:
            results['P(D)'] = P_D
            P_not_D = 1 - P_D
            results['P(¬D)'] = P_not_D

        if P_F is not None:
            results['P(F)'] = P_F
            P_not_F = 1 - P_F
            results['P(¬F)'] = P_not_F

        # Calculate P(B) using the law of total probability if P(B|A), P(B|¬A), and P(A) are given
        if P_B is None and P_B_given_A is not None and P_A is not None and P_B_given_not_A is not None:
            P_B = P_B_given_A * P_A + P_B_given_not_A * P_not_A
            results['P(B)'] = P_B
            P_not_B = 1 - P_B
            results['P(¬B)'] = P_not_B

        # Calculate P(A) using the law of total probability if P(A|B), P(A|¬B), and P(B) are given
        if P_A is None and P_A_given_B is not None and P_B is not None and P_A_given_not_B is not None:
            P_A = P_A_given_B * P_B + P_A_given_not_B * P_not_B
            results['P(A)'] = P_A
            P_not_A = 1 - P_A
            results['P(¬A)'] = P_not_A

        # Calculate P(A|B) using Bayes' theorem if P(A), P(B|A), and P(B) are given
        if P_A_given_B is None and P_B is not None and P_A is not None and P_B_given_A is not None:
            P_A_given_B = (P_B_given_A * P_A) / P_B
            results['P(A|B)'] = P_A_given_B

        # Calculate P(B|A) using Bayes' theorem if P(A|B), P(B), and P(A) are given
        if P_B_given_A is None and P_A_given_B is not None and P_A is not None and P_B is not None:
            P_B_given_A = (P_A_given_B * P_B) / P_A
            results['P(B|A)'] = P_B_given_A

        # Calculate P(B|¬A) if possible
        if P_B_given_not_A is None and P_B is not None and P_A is not None and P_B_given_A is not None:
            P_B_given_not_A = (P_B - P_B_given_A * P_A) / P_not_A
            results['P(B|¬A)'] = P_B_given_not_A

        if P_C_given_A is not None and P_A is not None:
            results['P(C|A)'] = P_C_given_A

        if P_A_given_C is not None and P_C is not None:
            if P_A is None:
                P_A = P_A_given_C * P_C
                results['P(A)'] = P_A
            if P_C_given_A is None:
                P_C_given_A = P_A_given_C * P_C / P_A
                results['P(C|A)'] = P_C_given_A

        if P_D_given_A is not None and P_A is not None:
            results['P(D|A)'] = P_D_given_A

        if P_A_given_D is not None and P_D is not None:
            if P_A is None:
                P_A = P_A_given_D * P_D
                results['P(A)'] = P_A
            if P_D_given_A is None:
                P_D_given_A = P_A_given_D * P_D / P_A
                results['P(D|A)'] = P_D_given_A

        if P_F_given_A is not None and P_A is not None:
            results['P(F|A)'] = P_F_given_A

        if P_A_given_F is not None and P_F is not None:
            if P_A is None:
                P_A = P_A_given_F * P_F
                results['P(A)'] = P_A
            if P_F_given_A is None:
                P_F_given_A = P_A_given_F * P_F / P_A
                results['P(F|A)'] = P_F_given_A

        if P_C_given_B is not None and P_B is not None:
            results['P(C|B)'] = P_C_given_B

        if P_B_given_C is not None and P_C is not None:
            results['P(B|C)'] = P_B_given_C

        if P_D_given_C is not None and P_C is not None:
            if P_C_given_D is None and P_D is not None:
                P_C_given_D = P_D_given_C * P_C / P_D
                results['P(C|D)'] = P_C_given_D

        if P_D_given_B is not None and P_B is not None:
            if P_C_given_D is None and P_D is not None:
                P_C_given_D = P_D_given_B * P_B / P_D
                results['P(C|D)'] = P_C_given_D

        if P_D_given_F is not None and P_F is not None:
            if P_C_given_D is None and P_D is not None:
                P_C_given_D = P_D_given_F * P_F / P_D
                results['P(C|D)'] = P_C_given_D

        if P_D_given_C is not None and P_D_given_B is not None and P_D_given_F is not None and P_C is not None and P_B is not None and P_F is not None:
            P_D = P_D_given_C * P_C + P_D_given_B * P_B + P_D_given_F * P_F
            results['P(D)'] = P_D

        # Handle new transportation parameters
        if P_D_given_F is not None and P_F is not None and P_D is not None:
            P_F_given_D = (P_D_given_F * P_F) / (
                (P_D_given_F * P_F) +
                (P_D_given_B * P_B if P_D_given_B is not None and P_B is not None else 0) +
                (P_D_given_C * P_C if P_D_given_C is not None and P_C is not None else 0)
            )
            results['P(F|D)'] = P_F_given_D

        # Populate remaining known parameters
        for key, value in self.parameters.items():
            results[key] = value

        return results

    def print_probabilities(self):
        results = self.calculate_all_probabilities()
        for key, value in results.items():
            print(f"{key}: {self.format_probability(value)}")
    
    def format_probability(self, probability):
        return f"{probability:.4f}" if probability is not None else "N/A"

# Example usage
if __name__ == "__main__":
    # Create an instance of BayesCalculator
    bc = BayesCalculator()
    
    # Set parameters for the initial scenario
    bc.set_parameter('P(A)', 0.5938)  # Probability that a country is from Africa
    bc.set_parameter('P(B|A)', 0.6316)  # Probability that a country has a GNP > 1000 USD given it is from Africa
    bc.set_parameter('P(B|¬A)', 0.1538)  # Probability that a country has a GNP > 1000 USD given it is not from Africa

#     # Uncomment to add more parameters if available
#     bc.set_parameter('P(A|¬B)', 0.03)  # P(A|¬B)
#     bc.set_parameter('P(A|B)', 0.97)  # P(A|B)
#     bc.set_parameter('P(B)', 0.5)  # P(B)
#     bc.set_parameter('P(¬B)', 0.99)  # P(¬B)
#     bc.set_parameter('P(C)', 0.1)  # P(C)
#     bc.set_parameter('P(D)', 0.6)  # P(D)
#     bc.set_parameter('P(A|C)', 0.04)  # P(A|C)
#     bc.set_parameter('P(A|D)', 0.03)  # P(A|D)
#     bc.set_parameter('P(F|A)', 0.6)  # P(F|A)
#     bc.set_parameter('P(B|A)', 0.2)  # P(B|A)
#     bc.set_parameter('P(C|A)', 1/5)  # P(C|A)
#     bc.set_parameter('P(A|C)', 0.5)  # P(A|C)
#     bc.set_parameter('P(C|B)', 0.4)  # P(C|B)
#     bc.set_parameter('P(B|C)', 0.6)  # P(B|C)
#     bc.set_parameter('P(D|A)', 2/5)  # P(D|A)
#     bc.set_parameter('P(F|A)', 2/5)  # P(F|A)
    # Print all probabilities
    bc.print_probabilities()
