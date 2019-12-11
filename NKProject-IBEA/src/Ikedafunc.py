import array
import sys
import random
import json
import subprocess
import pandas

import numpy
import scipy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import hypervolume
import src.emo as emo
import copy
from deap import creator
from deap import tools
from src.hv import HyperVolume
from src.objectives import Objectives
from deap import cma
import functools

# Problem size
N = 2
obj = 2
eps = 1e-10

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("IndividualMin", list, fitness=creator.FitnessMin, wholeFitness=None, volViolation=0, valConstr=None,
               volOverBounds=0, isFeasible=True, paretoRank=0)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("IndividualMax", list, fitness=creator.FitnessMax, wholeFitness=None, volViolation=0, valConstr=None,
               volOverBounds=0, isFeasible=True, paretoRank=0)


def zdt1(LOWBOUNDS, UPBOUNDS, hof, phase, ind):
    gnm = []
    conresult = []
    ref = [300, 300]

    for i, num in enumerate(ind):
        if num < LOWBOUNDS[i] - eps:
            gnm.append(LOWBOUNDS[i])
            ind.volViolation += numpy.abs(num - LOWBOUNDS[i])
        elif num > UPBOUNDS[i] + eps:
            gnm.append(UPBOUNDS[i])
            ind.volViolation += numpy.abs(UPBOUNDS[i] - num)
        else:
            gnm.append((trueUP[i] - trueLOW[i]) * (num / subD) + trueLOW[i])
            conresult.append(numpy.minimum(num - LOWBOUNDS[i], UPBOUNDS[i] - num))
    gnm = numpy.array(gnm)
    obj1 = Objectives.ikedaobj1(gnm)
    obj2 = Objectives.ikedaobj2(gnm)
    ind.wholeFitness = (-obj1, -obj2)
    for i in range(1, len(gnm)):
        conresult.append(gnm[0] - gnm[i])
    if phase == 0:
        result = obj1
    elif phase == 1:
        result = obj2
    else:
        cand = creator.IndividualMax(gnm)
        cand.wholeFitness = [obj1, obj2]
        newset = copy.deepcopy(hof)
        newset.append(cand)
        result = recHV(newset, ref)[1] - recHV(hof, ref)[1]
    ind.valConstr = conresult

    for i in conresult:
        if i < 0:
            ind.volViolation += abs(i)
    return result,


subD = 1
toolbox = base.Toolbox()


def loadBoundary():
    data = numpy.loadtxt('boundary.csv', delimiter=',', dtype=float)
    dlow = []
    dhigh = []
    for row in data:
        dlow.append(row[0])
        dhigh.append(row[1])
    return dlow, dhigh


def recHV(population, refs):
    truefront = emo.selFinal(population, 200)
    if len(truefront) == 1 and not truefront[0].isFeasible:
        return truefront, 0
    else:
        dcfront = copy.deepcopy(truefront)
        tfPoint = []
        for i, ind in enumerate(dcfront):
            if not checkDuplication(ind, dcfront[:i]):
                tfPoint.append([ind.wholeFitness[i] for i in range(len(ind.wholeFitness))])
        hy = HyperVolume(refs)
        HV = hy.compute(tfPoint)
        return truefront, HV


def checkDuplication(ind, front):
    for prep in front:
        count = 0
        for i in range(0, len(ind.fitness.values)):
            if str(prep.wholeFitness[i])[:16] == str(ind.wholeFitness[i])[:16]:
                count += 1
        if count == len(ind.fitness.values):
            return True
    return False


toolbox.register("evaluate", zdt1)


def main():
    # The cma module uses the numpy random number generator
    # ToDO:Implement restart strategy
    global trueUP, trueLOW
    trueLOW, trueUP = loadBoundary()
    numpy.random.seed()
    LOWBOUNDS = numpy.zeros(N)
    UPBOUNDS = numpy.ones(N)
    num_solutions = 30
    keeped_solution = []

    NGEN = 300
    eval_log = numpy.empty((0, 2), float)
    verbose = True
    create_plot = True
    indlogs = list()
    trueParetoLog = []
    allTF = []
    HVlog = []

    ref_cal = [1,1]
    sigmas = []
    indAses = []
    detA = []
    sucrate = []
    indfirst_pc = []

    for phase in range(0, num_solutions):
        # The MO-CMA-ES algorithm takes a full population as argument
        C = numpy.random.uniform(LOWBOUNDS, UPBOUNDS, N)
        strategy = cma.Strategy(C, 0.1)
        if (phase < obj):
            population = strategy.generate(creator.IndividualMin)
        else:
            population = strategy.generate(creator.IndividualMax)
        fitnesses = toolbox.map(functools.partial(toolbox.evaluate, LOWBOUNDS, UPBOUNDS, keeped_solution,  phase),
                                population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
            if ind.volViolation > 0:
                if phase < obj:
                    ind.fitness.values = ind.fitness.values[0] + ind.volViolation * 1000,
                else:
                    ind.fitness.values = (-ind.fitness.values[0] - ind.volViolation * 1000,)
                ind.isFeasible = False
            eval_log = numpy.append(eval_log, numpy.array([fit[0]]))
            genom = [ind[i] for i in range(0, N)]
            if ind.isFeasible:
                indlogs.append([genom, fit, 1, ind.valConstr, 0])
            else:
                indlogs.append([genom, fit, 0, ind.valConstr, 0])

        strategy.update(population)
        hof = tools.HallOfFame(1)
        t0, h0 = recHV(population, ref_cal)
        for ind in t0:
            geno = [ind[i] for i in range(0, N)]
            if ind.isFeasible:
                allTF.append([geno, ind.fitness.values, 1, ind.valConstr, 0])
            else:
                allTF.append([geno, ind.fitness.values, 0, ind.valConstr, 0])
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        logbook = tools.Logbook()

        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
        for gen in range(NGEN):
            # Generate a new population
            if (phase < obj):
                population = strategy.generate(creator.IndividualMin)
            else:
                population = strategy.generate(creator.IndividualMax)

            # Evaluate the individuals
            fitnesses = toolbox.map(
                functools.partial(toolbox.evaluate, LOWBOUNDS, UPBOUNDS, keeped_solution, phase), population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
                if ind.volViolation > 0:
                    if phase < obj:
                        ind.fitness.values = ind.fitness.values[0] + ind.volViolation * 1000,
                    else:
                        ind.fitness.values = (-ind.fitness.values[0] - ind.volViolation * 1000,)
                    ind.isFeasible = False
                eval_log = numpy.append(eval_log, numpy.array([fit[0]]))
                genom = [ind[i] for i in range(0, N)]
                if ind.isFeasible:
                    indlogs.append([genom, fit, 1, ind.valConstr, gen])
                else:
                    indlogs.append([genom, fit, 0, ind.valConstr, gen])

            # Update the strategy with the evaluated individuals
            hof.update(population)
            strategy.update(population)
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(population), **record)
            if verbose:
                print(logbook.stream)
        keeped_solution.append(hof[0])
        _ , he = recHV(keeped_solution, ref_cal)
        HVlog.append(he)
        print("Current population hypervolume is " + str(he))


    if verbose:
        print("Final population hypervolume is " + str(HVlog[-1]))
        # Note that we use a penalty to guide the search to feasible solutions,

        # but there is no guarantee that individuals are valid.

        # We expect the best individuals will be within bounds or very close.
        print("Final solutions:")
        print(numpy.asarray(keeped_solution))

    trIndLog = transLogs(indlogs)
    df = pandas.DataFrame(trIndLog)

    return trueParetoLog[-1], eval_log, df, HVlog, allTF, sigmas, indAses, sucrate, indfirst_pc, detA


def transLogs(logs):
    trdlogs = list()
    for ind in logs:
        log = []
        for i in range(0, len(ind)):
            if isinstance(ind[i], list) or isinstance(ind[i], tuple):
                for j in range(0, len(ind[i])):
                    log.append(ind[i][j])
            else:
                log.append(ind[i])
        trdlogs.append(log)
    return trdlogs


def makeFile(solutions, fitness_history, df, HVlog, tflog, siglog, domiAlog, indAlog, sucrate, domifirstPC, indfirstPC,
             detA, reps):
    df.to_csv("MONES" + "%03.f" % (reps) + ".csv")
    dh = pandas.DataFrame(HVlog)
    dh.to_csv("MONESHV" + "%03.f" % (reps) + ".csv")
    itlog = transLogs(tflog)
    dtr = pandas.DataFrame(itlog)
    dtr.to_csv("MONESTF" + "%03.f" % (reps) + ".csv")
    sigtr = transLogs(siglog)
    dsig = pandas.DataFrame(sigtr)
    dsig.to_csv("NESSigmaLog" + "%03.f" % (reps) + ".csv")
    domiAtr = transLogs(domiAlog)
    domiAtr = pandas.DataFrame(domiAtr)
    domiAtr.to_csv("NESdomiALog" + "%03.f" % (reps) + ".csv")
    indAtr = transLogs(indAlog)
    indAd = pandas.DataFrame(indAtr)
    indAd.to_csv("NESindALog" + "%03.f" % (reps) + ".csv")
    tsucrate = transLogs(sucrate)
    tsucrated = pandas.DataFrame(tsucrate)
    tsucrated.to_csv("NESsuccessRateLog" + "%03.f" % (reps) + ".csv")
    domitfpc = transLogs(domifirstPC)
    domittfpc = pandas.DataFrame(domitfpc)
    domittfpc.to_csv("NESdomiFirstPC" + "%03.f" % (reps) + ".csv")
    indtfpc = transLogs(indfirstPC)
    indttfpc = pandas.DataFrame(indtfpc)
    indttfpc.to_csv("NESindFirstPC" + "%03.f" % (reps) + ".csv")
    detAd = pandas.DataFrame(detA)
    detAd.to_csv("NESdetALog" + "%03.f" % (reps) + ".csv")


if __name__ == "__main__":
    num_exc = 40
    for reps in range(0, num_exc):
        solutions, fitness_history, df, HVlog, tflog, siglog, domiAlog, indAlog, sucrate, domifirstPC, indfirstPC, detA = main()

        fig = plt.figure()
        plt.title("Multi-objective minimization via I-IBEA-NES")
        plt.xlabel("f1")
        plt.ylabel("f2")
        # Limit the scale because our history values include the penalty.
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.grid(True)
        # Plot all history. Note the values include the penalty.
        fitness_history = numpy.asarray(fitness_history)
        plt.scatter(fitness_history[:, 0], fitness_history[:, 1], facecolors='none', edgecolors="lightblue", s=6)
        valid_front = numpy.array([ind.fitness.values for ind in solutions])
        plt.scatter(valid_front[:, 0], valid_front[:, 1], c="g", s=6)
        print("Writing cma_mo.png")
        plt.savefig("I-IBEA-NES" + "%03.f" % (reps) + ".png", dpi=300)
        plt.close(fig)

        hvfig = plt.figure()
        plt.title("HV/Generation")
        plt.xlabel("Generation")
        plt.ylabel("HV")
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xlim((1, 2000))
        plt.ylim((0, 1))
        GEN = numpy.array([i for i in range(1, len(HVlog) + 1)])
        plt.plot(GEN, HVlog)
        plt.grid(True)
        plt.savefig("MONESHV" + "%03.f" % (reps) + ".png", dpi=300)
        plt.close(hvfig)
