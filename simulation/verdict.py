def get_undercut_verdict(simulation_results, driver_a, driver_b):
    """
    Evaluates the simulation results to return a plain-English verdict 
    for the earliest undercut window found.
    """
    # Find the first lap where undercut_window_open is True
    for res in simulation_results:
        if res['undercut_window_open']:
            first_open_lap = res['lap']
            return f"Undercut window opens on lap {first_open_lap}. {driver_b} can pit now and recover the pit stop loss within 3 laps on fresher tyres."
            
    # If no window is found (crossover <= 3), check if there is any crossover at all
    for res in simulation_results:
        if res['crossover_lap'] is not None:
            return "No immediate undercut threat. Pit stop loss takes more than 3 laps to recover \u2014 undercut risk is low this stint."
            
    # No crossover found
    return f"No undercut window detected in the next 15 laps. {driver_a} is safe to stay out."
