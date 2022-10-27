from flask import Flask, render_template, request, url_for, flash, redirect
from flask_socketio import SocketIO, emit
from flask_cors import CORS, cross_origin
import pandas as pd
import sqlite3
import statistics
import numpy as np
import random
import multiprocess as mp
import threading
from functools import reduce
import uuid
import time
import warnings 
warnings.filterwarnings('ignore')
from yahoofinancials import YahooFinancials
import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)
numTradeDays = 252
portfolioValue=1000
dbName = "stocks.db"
database = None
threads = []
print('number of processors: ', mp.cpu_count())
socketio = SocketIO(app, async_mode=None, cors_allowed_origins='*')
socketio.run(app)


def get_db_connection():
    print('in connection')
    conn = sqlite3.connect('stocks.db')
    conn.row_factory = sqlite3.Row
    return conn

def pullFromDatabase(tickers, startdate, enddate):
    print("In pullFromDatabase")
    con = sqlite3.connect(dbName)
    #con.row_factory = sqlite3.Row
    cur = con.cursor()
    sqlSelect = "SELECT ticker, date, close, dailyreturn FROM alltickers where ticker in ("
    for ticker in tickers:
        sqlSelect = sqlSelect+"\'" + ticker+"\',"
    sqlSelect = sqlSelect.strip(',')+ ") and date>=\'" + startdate + "\' and date<=\'" + enddate +"\'"
    #print(sqlSelect)
    res = cur.execute(sqlSelect)
    data = res.fetchall()
    cur.close()
    con.close()
    print("Data Fetched")
    #print(data)
    return data

def dateToTimestamp(datestr):
    element = datetime.datetime.strptime(datestr,"%Y-%m-%d")
    return datetime.datetime.timestamp(element)
def timestampToDate(timestamp):
    return time.strftime("%Y-%m-%d", time.localtime(int(timestamp)))  

def fetchPortfolio(tickers, startdate, enddate):
    dataSubset = database[database['ticker'].isin(tickers)]
    return dataSubset

#Method uses the daily returns for each ticker
def getAnnualizedWeightedAvgDailyReturn(portfolio, portfolioValues):
    tickers = portfolio.iloc[:, 0];
    tickersAlloc = portfolio.iloc[:,1];
    totalReturn = 0
    for i in range(len(tickers)): #[0.5,0.5]
        ticker = tickers[i]
        proportion = tickersAlloc[i]
        tickerValues = pd.DataFrame(portfolioValues[(portfolioValues.ticker==ticker)])
        meanDailyReturn = statistics.mean(tickerValues['dailyreturn'])
        totalReturn += meanDailyReturn * proportion
    return totalReturn * numTradeDays
        

def getStandardDeviationForTicker(portfolioValues, ticker): 
    tickerValues = portfolioValues[(portfolioValues.ticker==ticker)]
    stddev = statistics.stdev(tickerValues['dailyreturn'])
    return stddev

def getPortfolioStandardDeviation(portfolio, startdate, enddate):
    tickers = portfolio.iloc[:, 0];
    tickersAlloc = portfolio.iloc[:,1];
    portfolioValues = fetchPortfolio(tickers, startdate, enddate)
    
    part_1=0
    for t in range(len(tickers)): #[0.5,0.5]
        proportion = tickersAlloc[t]
        ticker = tickers[t]
        sd = getStandardDeviationForTicker(portfolioValues, ticker)
        pt1_ticker = np.multiply(proportion,sd)**2
        #print ('pt 1 ' , ticker, ' proportion: ', proportion, ' sd: ', sd, ' pt1: ', pt1_ticker)
        part_1 += pt1_ticker
    #print ('pt 1 summation ' , part_1)  
    
    part_2=0 #The covariance matrix
    cov_prep = portfolioValues.pivot(index='date', columns='ticker', values='dailyreturn')
    cov_matrix = pd.DataFrame.cov(cov_prep)
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            if (i==j):
                continue #assume covariance = 0 hence the term 2*w1*w2*sigma1*sigma2*cov(ticker1, ticker2) will be 0
            w1=tickersAlloc[i]
            w2=tickersAlloc[j]
            sigma1=getStandardDeviationForTicker(portfolioValues, tickers[i])
            sigma2=getStandardDeviationForTicker(portfolioValues, tickers[j])
            cov=cov_matrix[tickers[i]][tickers[j]]
            #print('pt 2 ', i,' ',j,': [w1]', w1, ', [w2]', w2, ', [sg1]', sigma1, ', [sg2]', sigma2, ', cov', cov)
            part_2 += w1*w2*sigma1*sigma2*cov
    #print ('pt 2 summation ' , part_2)  
    portfolioVariance = part_1+part_2
    return (portfolioVariance ** 0.5) * (numTradeDays**0.5)

#Annual Basis
def getRiskFreeRate():
    return 0.0697

def getSharpeRatio(totalReturnPercentage, riskFreeRate, stddev):
    return (totalReturnPercentage-riskFreeRate) / stddev


def computeFitnessValue(portfolio, startdate, enddate):
    tickers = portfolio.iloc[:, 0];
    portfolioValues = fetchPortfolio(tickers, startdate, enddate)
    totalRetDailyAnnualized = getAnnualizedWeightedAvgDailyReturn(portfolio, portfolioValues)
    stddevRetDailyAnnualized = getPortfolioStandardDeviation(portfolio, startdate, enddate)
    sharpeRatio = getSharpeRatio(totalRetDailyAnnualized, getRiskFreeRate(), stddevRetDailyAnnualized)
    return sharpeRatio

def computeFitnessValueAndAppend(portfolio, startdate, enddate):
    sharpeRatio = computeFitnessValue(portfolio, startdate, enddate)
    new_portfolio = pd.DataFrame(portfolio)
    new_portfolio['sharpe'] = sharpeRatio
    return new_portfolio
      
    
#isSingleRow = boolean denoting if we are only inserting 1 row of data
def insert(table_name, value_tuple, isSingleRow):
    #print(dbName)
    conn = sqlite3.connect(dbName)
    cur = conn.cursor()
    numargs=0
    if (isSingleRow==True):
        numargs = len(value_tuple)
    else:
        numargs = len(value_tuple[0])
    sql_insert = 'insert into ' +  table_name + ' values('
    if (numargs==0):
        return
    for i in range(numargs):
        sql_insert += '?'
        if (i==numargs-1):
            sql_insert+=')'
        else:
            sql_insert+=','
    # The executemany method will insert multiple rows into SQLite table.
    #print(sql_insert)
    #print(value_tuple)
    if (isSingleRow==True):
        cur.execute(sql_insert, value_tuple)
    else:
        cur.executemany(sql_insert, value_tuple)
    conn.commit()
    cur.close()
    conn.close()   
    
def generateRunId():
    return (str(uuid.uuid4()))[0:6]

# Chromosome Definition
def chromosome(n, totalstocks):
    ''' Generates set of random numbers whose sum is equal to 1
        Input: n = number of stocks we want to invest out of totalstocks. totalstocks = universe of investible stocks
        Output: Array of random numbers'''
    #print("In Chromosome") # n = 10, totalstocks = 2
    n = np.minimum(n,totalstocks)
    #print("n:", n)
    ch = np.random.rand(n)
    ch = ch/sum(ch)
    #print("ch : ", ch)
    #disperse ch across the available stocks
    portfolio = [0] * totalstocks
    i = 0
    #print("portfolio:",portfolio) #[0,0]
    while (i < n):
        #print("i in chromosome", i)
        #look for portfolio[index] == 0 to replace with chromosome
        index = np.random.randint(0, totalstocks)
        #print("index", index)
        #print("portfolio[index]", portfolio[index])
        if (portfolio[index] == 0):
            portfolio[index] = ch[i]
            i = i + 1
    #print("portfolio in chromosome",portfolio)
    return portfolio
    
# Generate population
def generatePopulation(pop_size, stocktickers, maxStocks):
    #print("In generatePopulation")
    totalstocks = len(stocktickers)
    print("totalStocks : ", totalstocks)
    population = np.array([chromosome(maxStocks, totalstocks) for _ in range(pop_size)])
    return population
    
def toPortfolioDataframe(stocktickers, chromosome):
    return pd.DataFrame({'ticker':stocktickers, 'alloc':chromosome})    
        
def selectElitePopulation(stocktickers, population, selectionRate, startdate, enddate, runId, epoch,reqId='',publishToFrontend=False):
    #print("In selectElitePopulation")
    pool = mp.Pool(mp.cpu_count())
    n_chromosomes = len(population)
    population_fitness = [] 
    x1 = len(stocktickers)
    #convert all chromosomes to portfolio dataframes
    populationDFs =[]
    for chromosome in population:
        populationDFs.append(toPortfolioDataframe(stocktickers, chromosome))
        
    pool_outputs = []
    for i in range(n_chromosomes):
        pool_outputs.append(pool.apply_async(computeFitnessValue, args=(populationDFs[i], startdate, enddate))) 
    pool.close()
    pool.join()
    
    for i in range(len(pool_outputs)):
        population_fitness.append(pool_outputs[i].get())
    population = np.insert(population, x1, population_fitness, axis=1)
    population = sorted(population,key = lambda x: x[x1],reverse=True)
    writeGAPopulation(enddate, runId, epoch, population)
    if (publishToFrontend==True):
    	sendGAEpochToFrontend(enddate, runId, epoch, stocktickers, population,reqId)
    percentage_elite_idx = int(np.floor(len(population)* selectionRate))
    population = np.delete(population, x1, axis=1)
    return population[:percentage_elite_idx]
    
def mutation(parent):
    #print("In Mutation")
    child=parent.copy()
    n=np.random.choice(range(len(parent)),2)
    while (n[0]==n[1] or (child[n[0]]==0.0 and child[[n[1]]==0.0])):
        n=np.random.choice(range(len(parent)),2)
    child[n[0]],child[n[1]]=child[n[1]],child[n[0]]
    return child    
        
def crossover(parent1, parent2, maxStocks=10):
    #print("In crossover")
    length = len(parent1)
    crossoverPt = np.random.randint(0, length)
    child = [0] * length
    #print('crossover Pt', crossoverPt)
    numStocks = 0
    stockLoc = []
    for i in range(length):
        if (i<crossoverPt):
            child[i] =  parent1[i]
        else: 
            child[i] =  parent2[i]
        if (child[i] > 0.0):
            numStocks+=1;
            stockLoc.append(i)
        
    if (numStocks > maxStocks):
        toDrop = numStocks - maxStocks
        for j in range(toDrop):
            dropInd = stockLoc[np.random.randint(0, len(stockLoc))]
            child[dropInd] = 0.0
    sumOfChild = sum(child)           
    if (sumOfChild > 0.0):
        child = [float(i)/sumOfChild for i in child]      
    return child
    
    
def nextGeneration(pop_size, elite, crossoverRate, mutationRate):
    #print("In nextGeneration")
    new_population=[]
    pool_outputs = []
    pool = mp.Pool(mp.cpu_count())
    # replicate the elite population into the new population
    # mutate and/or crossover as necessary
    i=0
    while len(new_population) < pop_size:
        new_population.append(elite[i])
        i += 1
        if (i>len(elite)-1):
            i=0
    for j in range(1,pop_size):
        pool_outputs.append(pool.apply_async(mutateOrCrossover, args=(j, new_population, crossoverRate, mutationRate))) 
    pool.close()
    pool.join()
    new_population=[new_population[0]]
    for i in range(len(pool_outputs)):
        new_population.append(pool_outputs[i].get())
    return new_population


def mutateOrCrossover(j, new_population, crossoverRate, mutationRate):
    #print("In mutateOrCrossover")
    crossoverRandom = random.random()
    pop_size = len(new_population)
    sample = new_population[j]
    if (crossoverRandom <= crossoverRate):
        parent1 = sample
        p2Index = random.randint(0, pop_size-1)
        parent2 = new_population[p2Index]
        sample = crossover(parent1, parent2)
    mutateRandom = random.random()
    if (mutateRandom <= mutationRate):
        sample = mutation(new_population[j])
    return sample
    
def writeGAHyperparameters(runId, pop_size, stocktickers, maxStocks, maxIterations, selectionRate, crossoverRate, mutationRate, 
          startdate, enddate):
    value_tuple = (enddate, runId, startdate, enddate, str(stocktickers), maxStocks, maxIterations, selectionRate, mutationRate, crossoverRate)
    #print(value_tuple)
    insert('gahyperparams', value_tuple, True)

def writeGAResults(date, runId, epoch, chromosome, sharpe, risk,stockName):
    returns = sharpe * risk
    value_tuple = (date, runId, epoch, str(chromosome), sharpe, returns, risk,str(stockName))
    #print(value_tuple)
    insert('garesults', value_tuple, True)

def writeGAPopulation(date, runId, epoch, population):
    formatted_list = []
    populationLen = len(population[0])
    for i in range(len(population)):
        formatted_list.append((date, runId, epoch, i, str(population[i][0:populationLen-1]), population[i][populationLen-1:][0]))
    value_tuple = [tuple(item) for item in formatted_list]
    insert('gapopulation', value_tuple, False)

def sendGAHyperparametersToFrontend(runId, pop_size, stocktickers, maxStocks, maxIterations, selectionRate, crossoverRate, mutationRate, 
          startdate, enddate, reqId):
    data = {}
    data['type'] = 'gahyperparameters'
    data['msgid'] = reqId
    data['data'] = {}
    data['data']['date'] = enddate
    data['data']['runid'] = runId
    data['data']['startdate'] = startdate
    data['data']['enddate'] = enddate
    data['data']['popSize'] = pop_size
    data['data']['maxEpochs'] = maxIterations
    data['data']['selectionRate'] = selectionRate
    data['data']['mutationRate'] = mutationRate
    data['data']['crossoverRate'] = crossoverRate
    socketio.emit('my_response', data);
    
   
def sendGAEpochToFrontend(enddate, runId, epoch, stocktickers, population, reqId): 
    populationLen = len(population[0])
    data = {}
    data['type'] = 'gaEpoch'
    data['msgid'] = reqId
    data['runid'] = runId
    data['epoch'] = epoch
    data['stocktickers'] = stocktickers
    data['population'] = []
    for i in range(len(population)):
        memberChromosome = {}
        memberChromosome['popid'] = i
        memberChromosome['chromosome'] = []
        for j in range(populationLen-1):
            memberChromosome['chromosome'].append(population[i][j])
        memberChromosome['sharpe'] = population[i][populationLen-1:][0]
        data['population'].append(memberChromosome)
    socketio.emit('my_response', data);


def initialize(stocktickers, startdate, enddate, depth):
    global database
    if (depth==0):
        print(stocktickers)
        database = pd.DataFrame(pullFromDatabase(stocktickers, startdate, enddate))
        database.columns = ['ticker','date', 'price', 'dailyreturn']
        database.head()


def runGA(pop_size, stocktickers, maxStocks, maxIterations, selectionRate, crossoverRate, mutationRate, startdate, enddate, depth, maxDepth=5, runId='', reqId=''):
    initialize(stocktickers, startdate, enddate,depth)
    if (len(runId) == 0):
        runId = generateRunId()
    population = generatePopulation(pop_size, stocktickers, maxStocks)
    elite = selectElitePopulation(stocktickers, population, selectionRate, startdate, enddate, runId, 0, reqId, False)
    iteration=0 
    sharpeRatio = 0
    starttime = time.time()
    prevElite = elite
    distance=1000000
    convergeCount = 0
    writeGAHyperparameters(runId, pop_size, stocktickers, maxStocks, maxIterations, selectionRate, crossoverRate, mutationRate, 
          startdate, enddate)
    sendGAHyperparametersToFrontend(runId, pop_size, stocktickers, maxStocks, maxIterations, selectionRate, crossoverRate, mutationRate, 
          startdate, enddate, reqId)
    while (iteration < maxIterations):
        print('Iteration:',iteration)
        population = nextGeneration(pop_size,elite,crossoverRate,mutationRate)
        elite = selectElitePopulation(stocktickers, population, selectionRate, startdate, enddate, runId, iteration, reqId, True)
        #calculate Euclid dist between portfolios as a measure of similarity to determine
        #if GA has converged
        distance = np.linalg.norm(elite-prevElite)
        print(' distance ... ', distance, ' elite ', elite, ' prevElite ', prevElite)
        prevElite = elite
        topElite = toPortfolioDataframe(stocktickers, elite[0])
        sharpeRatio=computeFitnessValue(topElite, startdate, enddate)
        expectedRisk = getPortfolioStandardDeviation(topElite, startdate, enddate)
        print('Current Sharpe Ratio {} Expected Risk {}\n'.format(sharpeRatio, expectedRisk))
        #print('TopElite: ', topElite)
        if (distance < 0.0001):
            convergeCount+=1
        if (convergeCount==3):
            break
        iteration+=1
        

    if (sharpeRatio < 2): #Unsatisfactory local maximum! Rerun the whole GA with a different initial population
        print("UNSATISFACTORY SHARPE .... RERUN GA")
        depth += 1
        if (depth >= maxDepth):
           endtime = time.time()
           print('Portfolio of stocks creation after all the iterations finished.\n')
           print('total time taken: ', endtime-starttime)
           print("Depth : ", depth)
           stocks = topElite[topElite['alloc'] > 0]
           stockName = stocks['ticker'].values
           allocationPerc = stocks['alloc'].values
           print("stockName", stockName)
           print("allocationPerc", allocationPerc)
           writeGAResults(enddate, runId, iteration, allocationPerc, sharpeRatio, expectedRisk,stockName)
           threads.remove((reqId, threading.current_thread()))
           print("threads size ", len(threads))
           return stockName,allocationPerc 
        else:
           print("Running for Depth:", depth)
           runGA(pop_size, stocktickers, maxStocks, maxIterations, selectionRate, crossoverRate, mutationRate,startdate, enddate, depth, maxDepth, reqId=reqId)
    else:
        endtime = time.time()
        print('Portfolio of stocks creation after all the iterations finished.\n')
        print('total time taken: ', endtime-starttime)
        print("Depth : ", depth)
        stocks = topElite[topElite['alloc'] > 0]
        stockName = stocks['ticker'].values
        allocationPerc = stocks['alloc'].values
        print("stockName", stockName)
        print("allocationPerc", allocationPerc)
        writeGAResults(enddate, runId, iteration, allocationPerc, sharpeRatio, expectedRisk,stockName)
        threads.remove((reqId, threading.current_thread()))
        print("threads size ", len(threads))
        return stockName,allocationPerc
    
  
@app.route('/')
def hello_world():
    return("Hello World")        
        

    
@app.route('/index/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        stocktickers = []
        ticker1 = request.form['ticker1']
        stocktickers.append(ticker1)
        ticker2 = request.form['ticker2']
        stocktickers.append(ticker2)
        ticker3 = request.form['ticker3']
        stocktickers.append(ticker3)
        ticker4 = request.form['ticker4']
        stocktickers.append(ticker4)
        ticker5 = request.form['ticker5']
        stocktickers.append(ticker5)

        runGA(10, stocktickers, 10, 40, 0.3, 0.6, 0.6, '2021-09-27', '2022-10-05',0,1)
        #print(stockName,allocationPerc)
        #return render_template('index.html', stock = stockName, alloc = allocationPerc)
    return render_template('index.html')
    
    
@app.route('/customga', methods = ["POST"])
def customga():
    req = request.get_json(silent=False, force=True)
    reqId = req['reqId']
    customStocktickers = req['stocktickers']
    customPopSize = int(req['popSize'])
    customNumEpoch = int(req['numEpoch'])
    customSelectionRate = float(req['selectionRate'])
    customCrossoverRate = float(req['crossoverRate'])
    customMutationRate = float(req['mutationRate'])
    curtime = time.time()
    enddate = timestampToDate(curtime)
    startdate = timestampToDate(curtime-(24*60*60*365))
    depth = 0
    print('stocktickers ' , req['stocktickers'])
    print('popSize ' , req['popSize'])
    print('numEpoch ' , req['numEpoch'])
    print('selectionRate ' , req['selectionRate'])
    print('crossoverRate ' , req['crossoverRate'])
    print('mutationRate ' , req['mutationRate'])
    print('enddate ' , enddate)
    print('startdate ' , startdate)
    generatedRunId = generateRunId()
    threads.append((reqId, socketio.start_background_task(
    	target=runGA, 
    	pop_size=customPopSize, 
    	stocktickers=customStocktickers, 
    	maxStocks=len(customStocktickers), 
        maxIterations=customNumEpoch, 
        selectionRate=customSelectionRate, 
        crossoverRate=customCrossoverRate, 
        mutationRate=customMutationRate, 
        startdate=startdate, 
        enddate=enddate, 
        depth=depth,
        maxDepth=1,
        runId=generatedRunId,
        reqId=reqId
    )))
    print('=================number of threads ', len(threads))
    return 'Server received request for runid: ' + generatedRunId
    
