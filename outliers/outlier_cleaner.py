#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    tmp = [(age, net_worth, abs(prediction - net_worth)) for age, net_worth, prediction in zip(ages, net_worths, predictions)]
    return sorted(tmp, key=lambda x: x[2])[:9 * len(predictions) // 10]

    cleaned_data = []

    ### your code goes here

    
    return cleaned_data

