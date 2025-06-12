    # Print Solar results for example
    # time_range = pd.to_datetime(solar_model.dc_power_total[0:-1].index)
    # time_range = (time_range).strftime('%H:%M')
    # power = (solar_model.dc_power_total[0:-1].values)/1000
    # # Creating a DataFrame with 'Time (hours)' and 'Power'
    # df_solar_results = pd.DataFrame({
    #     'Time (hours)': time_range,
    #     'Solar Power (kW)': power
    # })

    # # Run basic model test
    # test_ev_charging(ev_model, charger_model, initial_charge=0.5, target_charge=0.9)

    # # Run test with Utility and Home
    # # ## Case 1
    # home_model.demand = 15 #(kW), EV call for 4.0 (kW)
    # test_ev_charging_v2(ev_model,charger_model,home_model,utility_model,initial_charge=0.7, target_charge=0.8, ev_call = 4)

    # ## Case 2
    # home_model.demand = 8 #(kW), EV call for 5.0 (kW) [Max]
    # test_ev_charging_v2(ev_model,charger_model,home_model,utility_model,initial_charge=0.5, target_charge=0.9, ev_call = ev_model.p_c_bar_ev)

    #Plot Solar Output on chosen day
    #test_solar_model(solar_model,dt)

    # Run test case with Utility, Home, and Solar
    #home_model.demand = 15 #(kW), EV call for 5.0 (kW) [Max]
    #test_ev_charging_v3(ev_model,charger_model,home_model,utility_model,solar_model,initial_charge_pre=0.8, initial_charge_post=0.6, target_charge= 1.0)

    # plot_obj_functions(x_b,x_ev,P_bat,P_ev,P_util,P_sol,P_dem,dt)
    # plot_inputs(P_sol,P_dem,dt,model_args.day)
    
    # start_weight = 1000
    # end_weight = 10000
    # increment = 500

    # for weight in range(start_weight, end_weight + 1, increment):
    #     logging.info(f"Running optimization for weight = {weight}")
    #     [x_b, x_ev, P_bat, P_ev, P_util, P_sol, P_dem] = evbm_optimization_v2(optimizer, weight)
        
    #     # You can also save or label plots by weight
    #     plot_results(x_b, x_ev, P_bat, P_ev, P_util, P_sol, P_dem, dt, model_args.day, weight)
    #     # plot_obj_functions(x_b, x_ev, P_bat, P_ev, P_util, P_sol, P_dem, dt)
    #     # plot_inputs(P_sol, P_dem, dt, model_args.day)