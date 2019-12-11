#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

#    Special thanks to Nikolaus Hansen for providing major part of
#    this code. The CMA-ES algorithm is provided in many other languages
#    and advanced versions at http://www.lri.fr/~hansen/cmaesintro.html.

"""A module that provides support for the Covariance Matrix Adaptation
Evolution Strategy.
"""
import copy
from math import sqrt, log, exp
import numpy
from scipy.linalg import expm
from scipy import dot, array, randn, eye, outer, exp, trace, floor, log, sqrt
import scipy

import functools
from operator import itemgetter
from operator import attrgetter
from deap import tools
from deap.tools._hypervolume import pyhv
import emo

class NaturalStrategyMultiObjective(object):
    def __init__(self, population, sigma, **params):
        self.parents = population
        self.dim = len(self.parents[0])

        # Selection
        self.mu = params.get("mu", len(self.parents))
        self.lambda_ = params.get("lambda_")
        #Learning rate
        self.etasigma = (3+numpy.log(self.dim))/(4+3*numpy.log(self.dim))/pow(self.dim,1.5)
        self.etaA = self.etasigma
        self.eps = numpy.sqrt(self.dim)*(1-1/(4*self.dim)+1/(21*numpy.power(self.dim,2)))
        self.infeasiblew = -1e-2

        # Internal parameters associated to the mu parent
        self.initdomiSigmas = sigma
        self.initindSigmas = sigma

        # counting sequential-achieving of infeasible
        self.infeasibleonind = 0
        self.infeasibleondom = 0
        self.thresholdinfeasible = 0

        #集計パラメータ
        self.dominating_Success = 0
        self.less_constraint = 0
        self.success_outer = 0
        self.success = 0
        self.missed_both_alive_out = 0
        self.missed_both_alive_in = 0
        self.parentonly_alive_out = 0
        self.parentonly_alive_in = 0

        self.indicator = params.get("indicator", tools.hypervolume)

    def generate(self, ind_init,a):
        """Generate a population of :math:`\lambda` individuals of type
        *ind_init* from the current strategy.

        :param ind_init: A function object that is able to initialize an
                         individual from a list.
        :returns: A list of individuals with a private attribute :attr:`_ps`.
                  This last attribute is essential to the update function, it
                  indicates that the individual is an offspring and the index
                  of its parent.
        """

        arz = numpy.random.randn(self.lambda_, self.dim)
        individuals = list()

        self.parents=sorted(self.parents, key=lambda x: x[0],reverse=True)
        # Make sure every parent has a parent tag and index
        for i, p in enumerate(self.parents):
            p._ps = "p", i
            p.Rank = 0

        for i in range(self.mu):
            if self.parents[i].dominateA is None:
                self.parents[i].dominateA = numpy.identity(self.dim)
                self.parents[i].indicatorA = numpy.identity(self.dim)
                self.parents[i].domisigma = self.initdomiSigmas
                self.parents[i].indsigma = self.initindSigmas
                self.parents[i].invA = numpy.identity(self.dim)
                self.parents[i].logdetA = 0
        # Each parent produce an offspring
        for i in range(self.lambda_):
            # print "Z", list(arz[i])


            cparent = copy.deepcopy(self.parents[i])
            individuals.append(ind_init(cparent + (1-a)*cparent.indsigma * numpy.dot(cparent.indicatorA, arz[i])+a*cparent.domisigma * numpy.dot(cparent.dominateA, arz[i])))
            individuals[i].theta = arz[i]
            individuals[i]._ps = "o", i
            individuals[i].Rank = 0
            individuals[i].contr = 0
            individuals[i].dominateA = cparent.dominateA
            individuals[i].indicatorA = cparent.indicatorA
            individuals[i].domisigma = cparent.domisigma
            individuals[i].indsigma = cparent.indsigma
            individuals[i].parent_genome = []
            individuals[i].parent_c = cparent.parent_c
            for mat in cparent:
                individuals[i].parent_genome.append(mat)
            individuals[i].parent_obj = cparent.fitness.values
        return individuals

    def _select(self, pop):
        """

        :type pop: list
        """
        candidates = self.parents + pop
        isPlenty = False
        if len(candidates) <= self.mu:
            return candidates, []

        pareto_fronts = emo.sortNondominated(candidates, len(candidates))
        if len(pareto_fronts[0]) > self.mu:
            isPlenty = True
        for i,front in enumerate(pareto_fronts):
            if len(front) == 1:
                for j in pareto_fronts[:i]:
                    if front[0]._ps[0] == "o":
                        pop[front[0]._ps[1]].Rank += len(j)
                    else:
                        self.parents[front[0]._ps[1]].Rank += len(j)
                if front[0]._ps[0] == "o":
                    pop[front[0]._ps[1]].Rank += 1
                    pop[front[0]._ps[1]].paretoRank = i + 1
                    pop[front[0]._ps[1]].contr = 0
                else:
                    self.parents[front[0]._ps[1]].Rank += 1
                    self.parents[front[0]._ps[1]].paretoRank = i + 1
                    self.parents[front[0]._ps[1]].contr = 0
            else:
                for m in range(0,len(front)):
                    for j in pareto_fronts[:i]:
                        if front[m]._ps[0] == "o":
                            pop[front[m]._ps[1]].Rank += len(j)
                        else:
                            self.parents[front[m]._ps[1]].Rank += len(j)
                wobjs = numpy.array([ind.fitness.wvalues for ind in front]) * -1
                refs = numpy.max(wobjs, axis=0) + 1

                def calContribution(idx, wobj, ref):
                    return pyhv.hypervolume(numpy.concatenate((wobj[:idx], wobj[idx + 1:])), ref)
                cont = list(map(functools.partial(calContribution,ref=refs,wobj=wobjs),list(range(len(front)))))
                for m in range(0,len(front)):
                    front[m].contr = cont[m]
                front.sort(key=attrgetter("contr"))
                for m in range(0,len(front)):
                    if front[m]._ps[0] == "o":
                        pop[front[m]._ps[1]].Rank += m+1
                        pop[front[m]._ps[1]].paretoRank = i + 1
                    else:
                        self.parents[front[m]._ps[1]].Rank += m+1
                        self.parents[front[m]._ps[1]].paretoRank = i + 1
        chosen = list()
        mid_front = None
        not_chosen = list()

        # Fill the next population (chosen) with the fronts until there is not enouch space
        # When an entire front does not fit in the space left we rely on the hypervolume
        # for this front
        # The remaining fronts are explicitely not chosen
        full = False
        for front in pareto_fronts:
            if len(chosen) + len(front) <= self.mu and not full:
                chosen += front
            elif mid_front is None and len(chosen) < self.mu:
                mid_front = front
                # With this front, we selected enough individuals
                full = True
            else:
                not_chosen += front

        # Separate the mid front to accept only k individuals
        k = self.mu - len(chosen)
        if k > 0:
            ref = numpy.array([ind.fitness.wvalues for ind in candidates]) * -1
            ref = numpy.max(ref, axis=0) + 1

            for _ in range(len(mid_front) - k):
                idx = self.indicator(mid_front, ref=ref)
                not_chosen.append(mid_front.pop(idx))
            chosen += mid_front
        return chosen, not_chosen,isPlenty



    def update(self, population,oddoreven):
        """Update the current covariance matrix strategies from the
        *population*.

        :param population: A list of individuals from which to update the
                           parameters.
        """

        chosen, not_chosen,isPlenty = self._select(population)
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        count6 = 0
        count7 = 0
        count8 = 0
        # Update the internal parameters for successful offspring


        # Only the offspring update the parameter set
        for ind in population:
            if ind.Rank < self.parents[ind._ps[1]].Rank and ind.Rank <= self.mu:
                gm = numpy.outer(ind.theta, ind.theta) - numpy.identity(self.dim)
                gsigma = numpy.trace(gm) / self.dim
                ga = gm - gsigma * numpy.identity(self.dim)
                proc = 0.5 * (self.etaA * ga)
                GGA = scipy.linalg.expm(proc)
                if gsigma > 0:
                    count1 += 1
                else:
                    count2 += 1
                if self.dominates(ind, self.parents[ind._ps[1]]):
                    count7 += 1
                    if oddoreven == 1:
                        self.infeasibleondom = 0
                        ind.domisigma = ind.domisigma * exp(self.etasigma * gsigma / 2.0)
                        ind.dominateA = numpy.dot(ind.dominateA, GGA)
                else:
                    if oddoreven == 0:
                        self.infeasibleonind = 0
                        ind.indsigma = ind.indsigma * exp(self.etasigma * gsigma / 2.0)
                        ind.indicatorA = numpy.dot(ind.indicatorA, GGA)

                if numpy.sum(ind.valConstr[1:]) < numpy.sum(self.parents[ind._ps[1]].valConstr[1:]):
                    count8 += 1

            elif ind.Rank > self.parents[ind._ps[1]].Rank and ind.Rank <= self.mu:
                gm = numpy.outer(ind.theta, ind.theta) - numpy.identity(self.dim)
                gsimga = numpy.trace(gm) / self.dim
                if gsimga > 0:
                    count3 += 1
                else:
                    count4 += 1
            elif ind.Rank > self.mu and self.parents[ind._ps[1]].Rank <= self.mu:
                gm = self.infeasiblew *( numpy.outer(ind.theta, ind.theta) - numpy.identity(self.dim))
                gsigma = numpy.trace(gm) / self.dim
                if numpy.trace(numpy.outer(ind.theta, ind.theta) - numpy.identity(self.dim)) > 0:
                    count5 += 1
                else:
                    count6 += 1
                ga = gm - gsigma * numpy.identity(self.dim)
                proc = 0.5 * (self.etaA * ga)
                GGA = scipy.linalg.expm(proc)
                if self.parents[ind._ps[1]].isFeasible and not ind.isFeasible:
                    if oddoreven == 0 and self.infeasibleonind < self.thresholdinfeasible:
                        self.infeasibleonind += 1
                    elif oddoreven == 1 and self.infeasibleondom < self.thresholdinfeasible:
                        self.infeasibleondom += 1
                    elif oddoreven == 0 and self.infeasibleonind >= self.thresholdinfeasible:
                        self.parents[ind._ps[1]].indsigma = self.parents[ind._ps[1]].indsigma * exp(self.etasigma * gsigma / 2.0)
                        self.parents[ind._ps[1]].indicatorA = numpy.dot(self.parents[ind._ps[1]].indicatorA, GGA)
                    elif oddoreven == 1 and self.infeasibleondom > self.thresholdinfeasible:
                        self.parents[ind._ps[1]].domisigma = self.parents[ind._ps[1]].domisigma * exp(self.etasigma * gsigma / 2.0)
                        self.parents[ind._ps[1]].dominateA = numpy.dot(self.parents[ind._ps[1]].dominateA, GGA)
            else:
                print(str(ind.Rank) + " and parent achieved " + str(self.parents[ind._ps[1]].Rank))

        self.dominating_Success = count7
        self.success_outer = count1
        self.success = count2
        self.missed_both_alive_out = count3
        self.missed_both_alive_out = count4
        self.parentonly_alive_out = count5
        self.parentonly_alive_in = count6
        self.less_constraint = count8
        self.parents = copy.deepcopy(chosen)

    def dominates(self,My, other, obj=slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.

        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        not_equal = False
        for self_wvalue, other_wvalue in zip(My.fitness.wvalues[obj], other.fitness.wvalues[obj]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False
        return not_equal