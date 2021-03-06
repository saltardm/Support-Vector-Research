# Machine Learning - Support Vector Regression (2.5:1 leverage limit)

# Uses scikit-learn machine learning to forecast the day's close price
#Finance jargon - going long is buying stock, going short is selling stock
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.svm import SVR
from sklearn import grid_search
from sklearn import preprocessing
from datetime import timedelta

def initialize(context):
    # Let's set a look up date inside our backtest to ensure we grab the correct security
    set_symbol_lookup_date('2015-05-04')
    
    # Use a very liquid set of stocks for quick order fills
    set_universe(universe.DollarVolumeUniverse(99.75, 100))
    context.stocks = symbols('SPY')
    set_benchmark(symbol('SPY'))
    #maintain 2.5 leverage buffer for contest rules
    context.entered_short = False
    context.leverage_buffer = 2.5
  
    
        
    # Perform forecast in the morning and take positions if needed. (try market close?)
    schedule_function(svm_trading, date_rules.every_day(), time_rules.market_open(minutes=5))
    #Only needed in testing/debugging to ensure orders are closed like in IB
    schedule_function(end_of_day, date_rules.every_day(), time_rules.market_close(minutes=1))
    
    # Use dicts to store items for plotting or comparison
    context.next_pred_price = {} # Current cycles prediction
    context.perform_flip = {}
    
    #Change us!
    context.history_len              = 45    # How many days in price history for training set
    context.stop_limit_percent       = 1  # If our cost_basis vs current price (percent diff) falls below this percentage, exit
#A stop limit is used to prevent against large losses during volatility swings
# Will be called on every trade event for the securities you specify. 
def handle_data(context, data):
    #Get EST Time every cycle, used for logging
    context.exchange_time  = pd.Timestamp(get_datetime()).tz_convert('US/Eastern')

    #Check that our portfolio does not  contain any invalid/external positions/securities
    check_invalid_positions(context, data)
    
    #Perform risk management
    risk_mgmt(context, data)
    
    # If needed, #Perform a desired flip now that the previous order is complete(intraday)
    for stock in data:
        if stock.symbol in context.perform_flip:
            if context.perform_flip[stock.symbol]:
                if check_if_no_conflicting_orders(stock):
                    #Perform a desired flip now that the previous order is complete
                    enter_position(context, data, stock)
                    context.perform_flip[stock.symbol] = False

    # Track the algorithm's leverage, and put it on the custom graph
    leverage = context.account.leverage
    
    record(leverage=leverage)
    
    # order_target functions don't consider open orders when making calculations.
    # Add this guard to prevent over-ordering
    if get_open_orders():
        return
    
    for stock in data:
        # Check the account leverage, leaving a buffer for open orders
        # Liquidate the short position if the leverage is approaching the 3x limit
        if leverage > context.leverage_buffer:
            log.info("Approaching leverage limit. Current leverage is %s" % (leverage))

            # Need to liquidate short position
            if context.entered_short == True:
                log.info("Liquidating position %s" % (stock))
                order_target_percent(stock, 0)
            return
        

def svm_trading(context, data):
    
    # Historical data, lets get the past days close prices for. 
    # +3(throw away todays partial data, use yesterday as test sample, offset training by 1 day for target values)
    history_open   = history(bar_count=context.history_len+3, frequency='1d', field='open_price')
    history_close  = history(bar_count=context.history_len+3, frequency='1d', field='close_price')
    history_high   = history(bar_count=context.history_len+3, frequency='1d', field='high')
    history_low    = history(bar_count=context.history_len+3, frequency='1d', field='low')
    history_volume = history(bar_count=context.history_len+3, frequency='1d', field='volume')

    # Make predictions on universe
    for stock in data:
        
        #Should only occur after this function is ran, so refresh it daily
        context.perform_flip[stock.symbol] = False
        
        # Make sure this stock has no existing orders or positions to simplify our portfolio handling.
        if check_if_no_conflicting_orders(stock):
            
            """ Data Configuration & Preprocessing """
            # What features does our model have? We can pick from the bar(open, close, high, low, volume)
            trainingVectors       = np.zeros((context.history_len, 5),dtype=np.float64)
            trainingVectors[:, 0] = np.array(history_open[stock].values)[:-3]
            trainingVectors[:, 1] = np.array(history_close[stock].values)[:-3]
            trainingVectors[:, 2] = np.array(history_high[stock].values)[:-3]
            trainingVectors[:, 3] = np.array(history_low[stock].values)[:-3]
            trainingVectors[:, 4] = np.array(history_volume[stock].values)[:-3]
            
            # Do a quick nan/inf check on all the features to ensure there is no bad data to crash the SVM
            if np.isnan(trainingVectors).any() or np.isinf(trainingVectors).any():
                log.debug("{0:s} had Nan or Inf in features, skipping for today.".format(stock.symbol))
                # Remove from dict to prevent actions on a skipped stock
                if stock.symbol in context.next_pred_price:
                    del context.next_pred_price[stock.symbol]
                #Continue the for loop to the next stock
                continue
            
            # create our scaling transformer to achieve a zero mean and unit variance(std=1). Scale the training data with it.
            scaler0               = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(trainingVectors[:, 0])
            scaler1               = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(trainingVectors[:, 1])
            scaler2               = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(trainingVectors[:, 2])
            scaler3               = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(trainingVectors[:, 3])
            scaler4               = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(trainingVectors[:, 4])
            
            # Apply the scale transform
            trainingVectors[:, 0] = scaler0.transform(trainingVectors[:, 0])
            trainingVectors[:, 1] = scaler1.transform(trainingVectors[:, 1])
            trainingVectors[:, 2] = scaler2.transform(trainingVectors[:, 2])
            trainingVectors[:, 3] = scaler3.transform(trainingVectors[:, 3])
            trainingVectors[:, 4] = scaler4.transform(trainingVectors[:, 4])

            # Target values, we want to use ^ yesterdays bar to predict this day's close price. Use close scaler????????
            targetValues          = np.zeros((context.history_len, ),dtype=np.float64)
            targetValues          = np.array(history_close[stock].values)[1:-2] - np.array(history_open[stock].values)[1:-2]
            scalerTarget          = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(targetValues)
            targetValues          = scalerTarget.transform(targetValues)
            
            # Test Samples, scaled using the feature training scaler
            testSamples           = np.zeros((1, 5), dtype=np.float64)
            testSamples[:, 0]     = np.array(history_open[stock].values)[-2]
            testSamples[:, 0]     = scaler0.transform(testSamples[:, 0])
            testSamples[:, 1]     = np.array(history_close[stock].values)[-2]
            testSamples[:, 1]     = scaler1.transform(testSamples[:, 1])
            testSamples[:, 2]     = np.array(history_high[stock].values)[-2]
            testSamples[:, 2]     = scaler2.transform(testSamples[:, 2])
            testSamples[:, 3]     = np.array(history_low[stock].values)[-2]
            testSamples[:, 3]     = scaler3.transform(testSamples[:, 3])
            testSamples[:, 4]     = np.array(history_volume[stock].values)[-2]
            testSamples[:, 4]     = scaler4.transform(testSamples[:, 4])
            
            """ Training Weight """
            weight_training = np.power(np.arange(1, targetValues.shape[0]+1,dtype=float), 1)/ \
                              np.power(np.arange(1, targetValues.shape[0]+1,dtype=float), 1).max()
            
            """ Model Optimization """
            parameters    = {'kernel':('linear', 'rbf'),'C':np.logspace(-2,1,13), 'gamma': np.logspace(-9, 1, 13)} #'kernel':('linear', 'rbf'),
            SVR_model     = SVR()
            clf           = grid_search.GridSearchCV(SVR_model, parameters)
            clf.fit(trainingVectors, targetValues)
            param_grid = dict(gamma=gamma_range, C=C_range)
            cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
            grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
            grid.fit(X, y)

            print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

            """ Forecast next close price """  
            SVR_model     = SVR(C=clf.best_params_["C"], gamma=clf.best_params_["gamma"]) #kernel=clf.best_params_["kernel"]
            SVR_model.fit(trainingVectors, targetValues, weight_training)
            y_predSVR     = scalerTarget.inverse_transform(SVR_model.predict(testSamples))[0]
            
            if len(data) == 1:
                record(regressionForecast = y_predSVR),Clog=np.log(clf.best_params_["C"]), Gamma = clf.best_params_["gamma"], Score=clf.best_score_)
            context.next_pred_price[stock.symbol] = y_predSVR

    # Count number of trades so we can split the availible cash properly
    context.number_of_trades_today = 0
    for stock in data:
        # Make sure this stock has no existing orders or positions to simplify our portfolio handling
        # Also check that we have a prediction stored in the dict
        if check_if_no_conflicting_orders(stock) and stock.symbol in context.next_pred_price:
                # If we plan to move on this stock, take count of it(explained more in actual buy statement below)(Make sure these match both buy statements.
                if  (context.next_pred_price[stock.symbol] > 0.0 and (context.portfolio.positions[stock.sid].amount == 0 or context.portfolio.positions[stock.sid].amount < 0)) or \
                    (context.next_pred_price[stock.symbol] < 0.0 and (context.portfolio.positions[stock.sid].amount == 0 or context.portfolio.positions[stock.sid].amount > 0)):
                    context.number_of_trades_today += 1
    #

    #Lets use record to plot how  many securities are traded on each day.       
    if len(data) >= 2:
        record(number_of_stocks_traded=context.number_of_trades_today)

    #Make buys and shorts based on forecasted direction of stock price
    for stock in data:
        # Make sure this stock has no existing orders or positions to simplify our portfolio handling
        # Also check that we have a prediction stored in the dict
        if check_if_no_conflicting_orders(stock) and stock.symbol in context.next_pred_price:
            
            if context.portfolio.positions[stock.sid].amount == 0:
                # We have no positons, buy, short, or stay
                if context.next_pred_price[stock.symbol] < 0.0 or context.next_pred_price[stock.symbol] == 1:
                    enter_position(context, data, stock)
                    
            elif context.portfolio.positions[stock.sid].amount > 0:
                # We are Long, short(exit then short) or stay
                if context.next_pred_price[stock.symbol] < 0.0:
                    exit_position(context, data, stock)
                    context.perform_flip[stock.symbol] = True
                
            elif context.portfolio.positions[stock.sid].amount < 0:
                # We are short, buy(Exit then buy) or stay
                if context.next_pred_price[stock.symbol] > 0.0:
                    exit_position(context, data, stock)
                    context.perform_flip[stock.symbol] = True
                    
def enter_position(context, data, stock):
    #Go long if we predict the close price will change more(upward) than our tollerance, 
    # apply same filter against current price vs predicted close in case of gap up/down.
    if context.next_pred_price[stock.symbol] > 0.0:
       #percent_change(context.next_pred_price[stock.symbol], data[stock]['price']) >= context.action_to_move_percent:

        # Place an order, and store the ID to fetch order info
        orderId    = order_target_percent(stock, 1.0/context.number_of_trades_today)
        # How many shares did we just order, since we used target percent of availible cash to place order not share count.
        shareCount = get_order(orderId).amount

        # We can add a timeout time on the order.
        #context.duration[orderId] = exchange_time + timedelta(minutes=5)

        # We need to calculate our own inter cycle portfolio snapshot as its not updated till next cycle.
        value_of_open_orders(context, data)
        availibleCash = context.portfolio.cash-context.cashCommitedToBuy-context.cashCommitedToSell

        log.info("+ BUY {0:,d} of {1:s} at ${2:,.2f} for ${3:,.2f} / ${4:,.2f} @ {5:d}:{6:d}"\
                 .format(shareCount,
                         stock.symbol,data[stock]['price'],
                         data[stock]['price']*shareCount, 
                         availibleCash,
                         context.exchange_time.hour,
                         context.exchange_time.minute))

    #Go short if we predict the close price will change more(downward) than our tollerance, 
    # apply same filter against current price vs predicted close incase of gap up/down.
    elif context.next_pred_price[stock.symbol] < 0.0:
         #percent_change(context.next_pred_price[stock.symbol], data[stock]['price']) <= -context.action_to_move_percent:

        #orderId    = order_target_percent(stock, -1.0/len(data))
        orderId    = order_target_percent(stock, -1.0/context.number_of_trades_today)
        # How many shares did we just order, since we used target percent of availible cash to place order not share count.
        shareCount = get_order(orderId).amount

        # We can add a timeout time on the order.
        #context.duration[orderId] = exchange_time + timedelta(minutes=5)

        # We need to calculate our own inter cycle portfolio snapshot as its not updated till next cycle.
        value_of_open_orders(context, data)
        availibleCash = context.portfolio.cash-context.cashCommitedToBuy+context.cashCommitedToSell

        log.info("- SHORT {0:,d} of {1:s} at ${2:,.2f} for ${3:,.2f} / ${4:,.2f} @ {5:d}:{6:d}"\
                 .format(shareCount,
                         stock.symbol,data[stock]['price'],
                         data[stock]['price']*shareCount, 
                         availibleCash,
                         context.exchange_time.hour,
                         context.exchange_time.minute))
        
def exit_position(context, data, stock):
    order_target(stock, 0.0)
    value_of_open_orders(context, data)
    availibleCash = context.portfolio.cash-context.cashCommitedToBuy-context.cashCommitedToSell
    log.info("- Exit {0:,d} of {1:s} at ${2:,.2f} for ${3:,.2f} / ${4:,.2f} @ {5:d}:{6:d}"\
                 .format(int(context.portfolio.positions[stock.sid].amount),
                         stock.symbol,
                         data[stock]['price'],
                         data[stock]['price']*context.portfolio.positions[stock.sid].amount,
                         availibleCash,
                         context.exchange_time.hour,
                         context.exchange_time.minute))
    
def risk_mgmt(context, data):
    if len(data) == 1:
        show_spacer = False
    
    # Dont rely on a single price point in case of noise, spike, or bottom of bar
    xminprice = history(bar_count=5, frequency='1m', field='open_price').mean()
    
    # Very limited Risk Management, please suggest more!
    
    
    # We are doing this every minute for every stock. Need to map or something for speed?
    for stock in data:
        # Make sure this stock has no existing orders or positions to simplify our portfolio handling. LONG Positions
        if check_if_no_conflicting_orders(stock) and context.portfolio.positions[stock.sid].amount > 0:
       
            if percent_change(xminprice[stock], context.portfolio.positions[stock.sid].cost_basis) <= -context.stop_limit_percent:
                order_target(stock, 0.0)
                value_of_open_orders(context, data)
                availibleCash = context.portfolio.cash-context.cashCommitedToBuy-context.cashCommitedToSell
                log.info("  ! SL-Exit {0:,d} of {1:s} at ${2:,.2f} for ${3:,.2f} / ${4:,.2f} @ {5:d}:{6:d}"\
                             .format(int(context.portfolio.positions[stock.sid].amount),
                                     stock.symbol,
                                     data[stock]['price'],
                                     data[stock]['price']*context.portfolio.positions[stock.sid].amount,
                                     availibleCash,
                                     context.exchange_time.hour,
                                     context.exchange_time.minute))
                if len(data) == 1:
                    show_spacer = True
                
        # Make sure this stock has no existing orders or positions to simplify our portfolio handling. SHORT Positions
        elif check_if_no_conflicting_orders(stock) and context.portfolio.positions[stock.sid].amount < 0:
            #Check the cost basis of our stock vs the current price, abandon if over the loss limit
            if percent_change(data[stock]['price'], context.portfolio.positions[stock.sid].cost_basis) >= context.stop_limit_percent:
                order_target(stock, 0.0)
                value_of_open_orders(context, data)
                availibleCash = context.portfolio.cash-context.cashCommitedToBuy-context.cashCommitedToSell
                log.info("  ! SL-Exit {0:,d} of {1:s} at ${2:,.2f} for ${3:,.2f} / ${4:,.2f} @ {5:d}:{6:d}"\
                             .format(int(context.portfolio.positions[stock.sid].amount),
                                     stock.symbol,
                                     data[stock]['price'],
                                     data[stock]['price']*context.portfolio.positions[stock.sid].amount,
                                     availibleCash,
                                     context.exchange_time.hour,
                                     context.exchange_time.minute))
                if len(data) == 1:
                    show_spacer = True
                
    if len(data) == 1:
        if show_spacer:
            log.info('') #This just gives us a space to make reading the 'daily' log sections more easily. 
            
                
#################################################################################################################################################

# Helper functions, allot of which is taken from the Quantopian documentation and forums(thanks Quantopian team for the great examples).
# Thread on these helper functions here: https://www.quantopian.com/posts/helper-functions-getting-started-on-quantopian


def check_if_no_conflicting_orders(stock):
    # Check that we are not already trying to move this stock
    open_orders = get_open_orders()
    safeToMove  = True
    if open_orders:
        for security, orders in open_orders.iteritems():
            for oo in orders:
                if oo.sid == stock.sid:
                    if oo.amount != 0:
                        safeToMove = False
    return safeToMove
    #

def check_invalid_positions(context, securities):
    # Check that the portfolio does not contain any broken positions
    # or external securities
    for sid, position in context.portfolio.positions.iteritems():
        if sid not in securities and position.amount != 0:
            errmsg = \
                "Invalid position found: {sid} amount = {amt} on {date}"\
                .format(sid=position.sid,
                        amt=position.amount,
                        date=get_datetime())
            raise Exception(errmsg)
            
def end_of_day(context, data):
   
    open_orders = get_open_orders()
    
    if open_orders:# or context.portfolio.positions_value > 0.:
        #log.info("")
        log.info("*** EOD: Stoping Orders & Printing Held ***")

    """# Print what positions we are holding overnight
    for stock in data:
        if context.portfolio.positions[stock.sid].amount != 0:
            log.info("{0:s} has remaining {1:,d} Positions worth ${2:,.2f}"\
                     .format(stock.symbol,
                             context.portfolio.positions[stock.sid].amount,
                             context.portfolio.positions[stock.sid].cost_basis\
                             *context.portfolio.positions[stock.sid].amount))"""
    # Cancle any open orders ourselves(In live trading this would be done for us, soon in backtest too)
    if open_orders:  
        # Cancle any open orders ourselves(In live trading this would be done for us, soon in backtest too)
        for security, orders in open_orders.iteritems():
            for oo in orders:
                log.info("X CANCLED {0:s} with {1:,d} / {2:,d} filled"\
                                     .format(security.symbol,
                                             oo.filled,
                                             oo.amount))
                cancel_order(oo)
    #

def fire_sale(context, data):
    # Sell everything in the portfolio, at market price
    show_spacer = False
    for stock in data:
        if context.portfolio.positions[stock.sid].amount != 0:
            order_target(stock, 0.0)
            value_of_open_orders(context, data)
            availibleCash = context.portfolio.cash-context.cashCommitedToBuy-context.cashCommitedToSell
            log.info("  * Exit {0:,d} of {1:s} at ${2:,.2f} for ${3:,.2f} / ${4:,.2f}  @ {5:d}:{6:d}"\
                         .format(int(context.portfolio.positions[stock.sid].amount),
                                 stock.symbol,
                                 data[stock]['price'],
                                 data[stock]['price']*context.portfolio.positions[stock.sid].amount,
                                 availibleCash,
                                 context.exchange_time.hour,
                                 context.exchange_time.minute))
            show_spacer = True
    if show_spacer:
        log.info('') #This just gives us a space to make reading the 'daily' log sections more easily 
    # 

def percent_change(new, old):
    return ((new-old)/old)*100.0
    
def value_of_open_orders(context, data):
    # Current cash commited to open orders, bit of an estimation for logging only
    context.currentCash = context.portfolio.cash
    open_orders = get_open_orders()
    context.cashCommitedToBuy  = 0.0
    context.cashCommitedToSell = 0.0
    if open_orders:
        for security, orders in open_orders.iteritems():
            for oo in orders:
                # Estimate value of existing order with current price, best to use order conditons?
                if(oo.amount>0):
                    context.cashCommitedToBuy  += oo.amount * data[oo.sid]['price']
                elif(oo.amount<0):
                    context.cashCommitedToSell += oo.amount * data[oo.sid]['price']
    
