#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2010-2019 German Aerospace Center (DLR) and others.
# This program and the accompanying materials
# are made available under the terms of the Eclipse Public License v2.0
# which accompanies this distribution, and is available at
# http://www.eclipse.org/legal/epl-v20.html
# SPDX-License-Identifier: EPL-2.0

# @file    randomTrips.py
# @author  Daniel Krajzewicz
# @author  Jakob Erdmann
# @author  Michael Behrisch
# @date    2010-03-06
# @version $Id$


from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import os
import sys
import random
import bisect
import subprocess
from collections import defaultdict
import math
import optparse

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import sumolib  # noqa
from sumolib.miscutils import euclidean  # noqa

DUAROUTER = sumolib.checkBinary('duarouter')

SOURCE_SUFFIX = ".src.xml"
SINK_SUFFIX = ".dst.xml"
VIA_SUFFIX = ".via.xml"


def get_options(args=None):
    optParser = optparse.OptionParser()
    optParser.add_option("--demand-duration", dest="demand_duration",
                         help="define the duration of a given demand")
    optParser.add_option("--lane-demand-variance", dest="lane_demand_variance",
                         help="define the net file (mandatory)")
    optParser.add_option("-n", "--net-file", dest="netfile",
                         help="define the net file (mandatory)")
    optParser.add_option("-a", "--additional-files", dest="additional",
                         help="define additional files to be loaded by the router")
    optParser.add_option("-o", "--output-trip-file", dest="tripfile",
                         default="trips.trips.xml", help="define the output trip filename")
    optParser.add_option("-r", "--route-file", dest="routefile",
                         help="generates route file with duarouter")
    optParser.add_option("--weights-prefix", dest="weightsprefix",
                         help="loads probabilities for being source, destination and via-edge from the files named " +
                         "<prefix>.src.xml, <prefix>.sink.xml and <prefix>.via.xml")
    optParser.add_option("--weights-output-prefix", dest="weights_outprefix",
                         help="generates weights files for visualisation")
    optParser.add_option("--pedestrians", action="store_true",
                         default=False, help="create a person file with pedestrian trips instead of vehicle trips")
    optParser.add_option("--persontrips", action="store_true",
                         default=False, help="create a person file with person trips instead of vehicle trips")
    optParser.add_option("--persontrip.transfer.car-walk", dest="carWalkMode",
                         help="Where are mode changes from car to walking allowed " +
                         "(possible values: 'ptStops', 'allJunctions' and combinations)")
    optParser.add_option("--persontrip.walkfactor", dest="walkfactor",
                         help="Use FLOAT as a factor on pedestrian maximum speed during intermodal routing")
    optParser.add_option("--prefix", dest="tripprefix",
                         default="", help="prefix for the trip ids")
    optParser.add_option("-t", "--trip-attributes", dest="tripattrs",
                         default="", help="additional trip attributes. When generating pedestrians, attributes for " +
                         "<person> and <walk> are supported.")
    optParser.add_option("--fringe-start-attributes", dest="fringeattrs",
                         default="", help="additional trip attributes when starting on a fringe.")
    optParser.add_option("-b", "--begin", type="float", default=0, help="begin time")
    optParser.add_option("-e", "--end", type="float", default=3600, help="end time (default 3600)")
    optParser.add_option(
        "-p", "--period", type="float", default=1, help="Generate vehicles with equidistant departure times and " +
        "period=FLOAT (default 1.0). If option --binomial is used, the expected arrival rate is set to 1/period.")
    optParser.add_option("-s", "--seed", type="int", dest='seed', help="random seed")
    optParser.add_option("-l", "--length", action="store_true",
                         default=False, help="weight edge probability by length")
    optParser.add_option("-L", "--lanes", action="store_true",
                         default=False, help="weight edge probability by number of lanes")
    optParser.add_option("--edge-param", dest="edgeParam",
                         help="use the given edge parameter as factor for edge")
    optParser.add_option("--speed-exponent", type="float", dest="speed_exponent",
                         default=0.0, help="weight edge probability by speed^<FLOAT> (default 0)")
    optParser.add_option("--fringe-factor", type="float", dest="fringe_factor",
                         default=1.0, help="multiply weight of fringe edges by <FLOAT> (default 1")
    optParser.add_option("--fringe-threshold", type="float", dest="fringe_threshold",
                         default=0.0, help="only consider edges with speed above <FLOAT> as fringe edges (default 0)")
    optParser.add_option("--allow-fringe", dest="allow_fringe", action="store_true",
                         default=False, help="Allow departing on edges that leave the network and arriving on edges " +
                         "that enter the network (via turnarounds or as 1-edge trips")
    optParser.add_option("--allow-fringe.min-length", type="float", dest="allow_fringe_min_length",
                         help="Allow departing on edges that leave the network and arriving on edges " +
                         "that enter the network, if they have at least the given length")
    optParser.add_option("--min-distance", type="float", dest="min_distance",
                         default=0.0, help="require start and end edges for each trip to be at least <FLOAT> m apart")
    optParser.add_option("--max-distance", type="float", dest="max_distance",
                         default=None, help="require start and end edges for each trip to be at most <FLOAT> m " +
                         "apart (default 0 which disables any checks)")
    optParser.add_option("-i", "--intermediate", type="int",
                         default=0, help="generates the given number of intermediate way points")
    optParser.add_option("--flows", type="int",
                         default=0, help="generates INT flows that together output vehicles with the specified period")
    optParser.add_option("--jtrrouter", action="store_true",
                         default=False, help="Create flows without destination as input for jtrrouter")
    optParser.add_option("--maxtries", type="int",
                         default=100, help="number of attemps for finding a trip which meets the distance constraints")
    optParser.add_option("--binomial", type="int", metavar="N",
                         help="If this is set, the number of departures per seconds will be drawn from a binomial " +
                         "distribution with n=N and p=PERIOD/N where PERIOD is the argument given to " +
                         "option --period. Tnumber of attemps for finding a trip which meets the distance constraints")
    optParser.add_option(
        "-c", "--vclass", "--edge-permission", default="passenger",
        help="only from and to edges which permit the given vehicle class")
    optParser.add_option(
        "--vehicle-class", help="The vehicle class assigned to the generated trips (adds a standard vType definition " +
        "to the output file).")
    optParser.add_option("--validate", default=False, action="store_true",
                         help="Whether to produce trip output that is already checked for connectivity")
    optParser.add_option("-v", "--verbose", action="store_true",
                         default=False, help="tell me what you are doing")
    (options, args) = optParser.parse_args(args=args)
    if not options.netfile:
        optParser.print_help()
        sys.exit(1)

    if options.persontrips:
        options.pedestrians = True

    if options.pedestrians:
        options.vclass = 'pedestrian'
        if options.flows > 0:
            print("Error: Person flows are not supported yet", file=sys.stderr)
            sys.exit(1)

    if options.validate and options.routefile is None:
        options.routefile = "routes.rou.xml"

    if options.period <= 0:
        print("Error: Period must be positive", file=sys.stderr)
        sys.exit(1)

    if options.jtrrouter and options.flows <= 0:
        print("Error: Option --jtrrouter must be used with option --flows", file=sys.stderr)
        sys.exit(1)

    if options.vehicle_class:
        if options.tripprefix:
            options.vtypeID = "%s_%s" % (options.tripprefix, options.vehicle_class)
        else:
            options.vtypeID = options.vehicle_class

        if 'type=' in options.tripattrs:
            print("Error: trip-attribute 'type' cannot be used together with option --vehicle-class", file=sys.stderr)
            sys.exit(1)

    return options


class InvalidGenerator(Exception):
    pass

# assigns a weight to each edge using weight_fun and then draws from a discrete
# distribution with these weights


class RandomEdgeGenerator:

    def __init__(self, lane_demand_variance, demand_duration, net, weight_fun):
        self.lane_demand_variance =  lane_demand_variance
        self.demand_duration = demand_duration
        self.net = net
        self.weight_fun = weight_fun
        self.cumulative_weights = []
        self.total_weight = 0
        #print("self.weight_fun", self.weight_fun)
        for edge in self.net._edges:
            # print edge.getID(), weight_fun(edge)
            self.total_weight += weight_fun(edge)
            self.cumulative_weights.append(self.total_weight)
        if self.total_weight == 0:
            raise InvalidGenerator()
            
    def update_probs(self):
        self.total_weight = 0
        self.cumulative_weights = []
        l = len(self.net._edges)
        #probs = random.rand
        w = []
        adjuster = np.absolute([int(random.gauss(0, int(self.lane_demand_variance))) for i in range(l)])
        #adjuster = adjuster-max(adjuster)
        #adjuster = np.exp(adjuster)/np.exp(adjuster).sum()
        
        #adjuster = np.absolute(self.env.r.normal(loc=0, scale = self.lane_demand_variance, size=l))
        #leaving_adjuster = np.absolute([random.gauss(0, 5) for i in range(l)])
        
        #entering_adjuster 
        #new_entering_edges_probs =  self.env.entering_edges_probs * (1+self.env.entering_adjuster) 
        #new_entering_edges_probs = self.env.new_entering_edges_probs -  self.env.new_entering_edges_probs.max()
        #new_leaving_edges_probs = self.env.leaving_edges_probs * (1+self.env.leaving_adjuster)
        #new_leaving_edges_probs = self.env.new_leaving_edges_probs - self.env.new_leaving_edges_probs.max()
        #self.env.new_entering_edges_probs = np.exp(self.env.new_entering_edges_probs)/np.exp(self.env.new_entering_edges_probs).sum()
        #self.env.new_leaving_edges_probs = np.exp(self.env.new_leaving_edges_probs)/np.exp(self.env.new_leaving_edges_probs).sum()              
        for idx, edge in enumerate(self.net._edges):
            #print("edge", edge)
            #print("self.wf(edge)", self.weight_fun(edge))
            w.append(self.weight_fun(edge) + adjuster[idx]) 
        
        w = w-max(w)
        w = np.exp(w)/np.exp(w).sum()
            
        #print("w",w)
        for w_ in w:
            self.total_weight += w_
            #self.total_weight += self.weight_fun(edge) 
            self.cumulative_weights.append(self.total_weight)
        #for edge in w_list:
            #self.total_weight += self.weight_fun(edge)
            #self.cumulative_weights.append(self.total_weight)
        #print("w_list", w_list)
    def get(self, depart = None):
        #print("depart", depart)
        if depart is not None:
            if int(depart+1) % int(self.demand_duration) ==0:
                #print("dd", self.demand_duration)
                #print("depart", depart)
                self.update_probs()
        #print("tw", self.cumulative_weights)
        #print("total w1", self.total_weight)
        #self.total_weight = int(sum(self.cumulative_weights))
        #print("total w2", self.total_weight)
        r = random.random() * self.total_weight
        #print('r', r)
        index = bisect.bisect(self.cumulative_weights, r)
        #print(self.net._edges[index])
        return self.net._edges[index]

    def write_weights(self, fname):
        # normalize to [0,100]
        normalizer = 100.0 / max(1, max(map(self.weight_fun, self.net._edges)))
        weights = [(self.weight_fun(e) * normalizer, e.getID()) for e in self.net.getEdges()]
        weights.sort(reverse=True)
        with open(fname, 'w+') as f:
            f.write('<edgedata>\n')
            f.write('    <interval begin="0" end="10">\n')
            for weight, edgeID in weights:
                f.write('        <edge id="%s" value="%0.2f"/>\n' %
                        (edgeID, weight))
            f.write('    </interval>\n')
            f.write('</edgedata>\n')


class RandomTripGenerator:

    def __init__(self, source_generator, sink_generator, via_generator, intermediate, pedestrians):
        self.source_generator = source_generator
        self.sink_generator = sink_generator
        self.via_generator = via_generator
        self.intermediate = intermediate
        self.pedestrians = pedestrians

    def get_trip(self, min_distance, max_distance, maxtries=100, depart = None):
        for i in range(maxtries):
            source_edge = self.source_generator.get(depart)
            intermediate = [self.via_generator.get(depart)
                            for i in range(self.intermediate)]
            sink_edge = self.sink_generator.get(depart)
            if self.pedestrians:
                destCoord = sink_edge.getFromNode().getCoord()
            else:
                destCoord = sink_edge.getToNode().getCoord()

            coords = ([source_edge.getFromNode().getCoord()] +
                      [e.getFromNode().getCoord() for e in intermediate] +
                      [destCoord])
            distance = sum([euclidean(p, q)
                            for p, q in zip(coords[:-1], coords[1:])])
            if distance >= min_distance and (max_distance is None or distance < max_distance):
                return source_edge, sink_edge, intermediate
        raise Exception("no trip found after %s tries" % maxtries)


def get_prob_fun(options, fringe_bonus, fringe_forbidden):
    # fringe_bonus None generates intermediate way points
    def edge_probability(edge):
        if options.vclass and not edge.allows(options.vclass):
            return 0  # not allowed
        if fringe_bonus is None and edge.is_fringe() and not options.pedestrians:
            return 0  # not suitable as intermediate way point
        if (fringe_forbidden is not None and edge.is_fringe(getattr(edge, fringe_forbidden)) and
                not options.pedestrians and
                (options.allow_fringe_min_length is None or edge.getLength() < options.allow_fringe_min_length)):
            return 0  # the wrong kind of fringe
        prob = 1
        if options.length:
            prob *= edge.getLength()
        if options.lanes:
            prob *= edge.getLaneNumber()
        prob *= (edge.getSpeed() ** options.speed_exponent)
        if (options.fringe_factor != 1.0 and
                not options.pedestrians and
                fringe_bonus is not None and
                edge.getSpeed() > options.fringe_threshold and
                edge.is_fringe(getattr(edge, fringe_bonus))):
            prob *= options.fringe_factor
        if options.edgeParam is not None:
            prob *= float(edge.getParam(options.edgeParam, 1.0))
        return prob
    return edge_probability


class LoadedProps:

    def __init__(self, fname):
        self.weights = defaultdict(lambda: 0)
        for edge in sumolib.output.parse_fast(fname, 'edge', ['id', 'value']):
            self.weights[edge.id] = float(edge.value)

    def __call__(self, edge):
        print("self.weights", self.weights)
        return self.weights[edge.getID()]


def buildTripGenerator(net, options):
    try:
        forbidden_source_fringe = None if options.allow_fringe else "_outgoing"
        forbidden_sink_fringe = None if options.allow_fringe else "_incoming"
        source_generator = RandomEdgeGenerator(options.lane_demand_variance, options.demand_duration, 
            net, get_prob_fun(options, "_incoming", forbidden_source_fringe))
        sink_generator = RandomEdgeGenerator(options.lane_demand_variance, options.demand_duration,
            net, get_prob_fun(options, "_outgoing", forbidden_sink_fringe))
        if options.weightsprefix:
            if os.path.isfile(options.weightsprefix + SOURCE_SUFFIX):
                source_generator = RandomEdgeGenerator(options.lane_demand_variance, options.demand_duration,
                    net, LoadedProps(options.weightsprefix + SOURCE_SUFFIX))
            if os.path.isfile(options.weightsprefix + SINK_SUFFIX):
                sink_generator = RandomEdgeGenerator(options.lane_demand_variance, options.demand_duration,
                    net, LoadedProps(options.weightsprefix + SINK_SUFFIX))
    except InvalidGenerator:
        print("Error: no valid edges for generating source or destination. Try using option --allow-fringe",
              file=sys.stderr)
        return None

    try:
        via_generator = RandomEdgeGenerator(options.lane_demand_variance, options.demand_duration,
            net, get_prob_fun(options, None, None))
        if options.weightsprefix and os.path.isfile(options.weightsprefix + VIA_SUFFIX):
            via_generator = RandomEdgeGenerator(options.lane_demand_variance, options.demand_duration,
                net, LoadedProps(options.weightsprefix + VIA_SUFFIX))
    except InvalidGenerator:
        if options.intermediate > 0:
            print(
                "Error: no valid edges for generating intermediate points", file=sys.stderr)
            return None
        else:
            via_generator = None

    return RandomTripGenerator(
        source_generator, sink_generator, via_generator, options.intermediate, options.pedestrians)


def is_walk_attribute(attr):
    for cand in ['arrivalPos', 'speed=', 'duration=', 'busStop=']:
        if cand in attr:
            return True
    return False


def is_persontrip_attribute(attr):
    for cand in ['vTypes', 'modes']:
        if cand in attr:
            return True
    return False


def is_person_attribute(attr):
    for cand in ['departPos', 'type']:
        if cand in attr:
            return True
    return False


def is_vehicle_attribute(attr):
    for cand in ['depart', 'arrival', 'line', 'Number', 'type']:
        if cand in attr:
            return True
    return False


def split_trip_attributes(tripattrs, pedestrians, hasType):
    # handle attribute values with a space
    # assume that no attribute value includes an '=' sign
    allattrs = []
    for a in tripattrs.split():
        if "=" in a:
            allattrs.append(a)
        else:
            if len(allattrs) == 0:
                print("Warning: invalid trip-attribute '%s'" % a)
            else:
                allattrs[-1] += ' ' + a

    # figure out which of the tripattrs belong to the <person> or <vehicle>,
    # which belong to the <vType> and which belong to the <walk> or <persontrip>
    vehicleattrs = []
    personattrs = []
    vtypeattrs = []
    otherattrs = []
    for a in allattrs:
        if pedestrians:
            if is_walk_attribute(a) or is_persontrip_attribute(a):
                otherattrs.append(a)
            elif is_person_attribute(a):
                personattrs.append(a)
            else:
                vtypeattrs.append(a)
        else:
            if is_vehicle_attribute(a):
                vehicleattrs.append(a)
            else:
                vtypeattrs.append(a)

    if not hasType:
        if pedestrians:
            personattrs += vtypeattrs
        else:
            vehicleattrs += vtypeattrs
        vtypeattrs = []

    return (prependSpace(' '.join(vtypeattrs)),
            prependSpace(' '.join(vehicleattrs)),
            prependSpace(' '.join(personattrs)),
            prependSpace(' '.join(otherattrs)))


def prependSpace(s):
    if len(s) == 0 or s[0] == " ":
        return s
    else:
        return " " + s


def main(options):
    #print('seed',options.seed)    
    #print('seed',options.seed)
    random.seed(options.seed)
    np.random.seed(options.seed)
    r = np.random.RandomState(options.seed)   

    net = sumolib.net.readNet(options.netfile)
    if options.min_distance > net.getBBoxDiameter() * (options.intermediate + 1):
        options.intermediate = int(
            math.ceil(options.min_distance / net.getBBoxDiameter())) - 1
        print(("Warning: setting number of intermediate waypoints to %s to achieve a minimum trip length of " +
               "%s in a network with diameter %.2f.") % (
            options.intermediate, options.min_distance, net.getBBoxDiameter()))

    trip_generator = buildTripGenerator(net, options)
    idx = 0

    vtypeattrs, options.tripattrs, personattrs, otherattrs = split_trip_attributes(
        options.tripattrs, options.pedestrians, options.vehicle_class)

    vias = {}

    def generate_one(idx, depart):
        label = "%s%s" % (options.tripprefix, idx)
        try:
            source_edge, sink_edge, intermediate = trip_generator.get_trip(
                options.min_distance, options.max_distance, options.maxtries, depart = depart)
            combined_attrs = options.tripattrs
            if options.fringeattrs and source_edge.is_fringe(source_edge._incoming):
                combined_attrs += " " + options.fringeattrs
            via = ""
            if len(intermediate) > 0:
                via = ' via="%s" ' % ' '.join(
                    [e.getID() for e in intermediate])
                if options.validate:
                    vias[label] = via
            if options.pedestrians:
                fouttrips.write(
                    '    <person id="%s" depart="%.2f"%s>\n' % (label, depart, personattrs))
                if options.persontrips:
                    fouttrips.write(
                        '        <personTrip from="%s" to="%s"%s/>\n' % (
                            source_edge.getID(), sink_edge.getID(), otherattrs))
                else:
                    fouttrips.write(
                        '        <walk from="%s" to="%s"%s/>\n' % (source_edge.getID(), sink_edge.getID(), otherattrs))
                fouttrips.write('    </person>\n')
            elif options.flows > 0:
                to = '' if options.jtrrouter else ' to="%s"' % sink_edge.getID()
                if options.binomial:
                    for j in range(options.binomial):
                        fouttrips.write(('    <flow id="%s#%s" begin="%s" end="%s" probability="%s" ' +
                                         'from="%s"%s%s%s/>\n') % (
                            label, j, options.begin, options.end, 1.0 / options.period / options.binomial,
                            source_edge.getID(), to, via, combined_attrs))
                else:
                    fouttrips.write(('    <flow id="%s" begin="%s" end="%s" period="%s" from="%s"%s%s%s/>\n') % (
                        label, options.begin, options.end, options.period * options.flows, source_edge.getID(),
                        to, via, combined_attrs))
            else:
                fouttrips.write('    <trip id="%s" depart="%.2f" from="%s" to="%s"%s%s/>\n' % (
                    label, depart, source_edge.getID(), sink_edge.getID(), via, combined_attrs))
        except Exception as exc:
            print(exc, file=sys.stderr)
        return idx + 1

    with open(options.tripfile, 'w') as fouttrips:
        sumolib.writeXMLHeader(fouttrips, "$Id$", "routes")  # noqa
        if options.vehicle_class:
            fouttrips.write('    <vType id="%s" vClass="%s"%s/>\n' %
                            (options.vtypeID, options.vehicle_class, vtypeattrs))
            options.tripattrs += ' type="%s"' % options.vtypeID
            personattrs += ' type="%s"' % options.vtypeID
        depart = options.begin
        if trip_generator:
            if options.flows == 0:
                while depart < options.end:
                    #print("idx", idx)
                    #print("depart", depart)
                    if options.binomial is None:
                        # generate with constant spacing
                        idx = generate_one(idx, depart)
                        depart += options.period
                    else:
                        # draw n times from a Bernoulli distribution
                        # for an average arrival rate of 1 / period
                        prob = 1.0 / options.period / options.binomial
                        for i in range(options.binomial):
                            if random.random() < prob:
                                idx = generate_one(idx, depart)
                        depart += 1
            else:
                for i in range(options.flows):
                    idx = generate_one(idx, depart)

        fouttrips.write("</routes>\n")

    # call duarouter for routes or validated trips
    args = [DUAROUTER, '-n', options.netfile, '-r', options.tripfile, '--ignore-errors',
            '--begin', str(options.begin), '--end', str(options.end), '--no-step-log', '--no-warnings']
    if options.additional is not None:
        args += ['--additional-files', options.additional]
    if options.carWalkMode is not None:
        args += ['--persontrip.transfer.car-walk', options.carWalkMode]
    if options.walkfactor is not None:
        args += ['--persontrip.walkfactor', options.walkfactor]
    if options.routefile:
        args2 = args + ['-o', options.routefile]
        print("calling ", " ".join(args2))
        subprocess.call(args2)

    if options.validate:
        # write to temporary file because the input is read incrementally
        tmpTrips = options.tripfile + ".tmp"
        args2 = args + ['-o', tmpTrips, '--write-trips']
        print("calling ", " ".join(args2))
        subprocess.call(args2)
        os.remove(options.tripfile)  # on windows, rename does not overwrite
        os.rename(tmpTrips, options.tripfile)

    if options.weights_outprefix:
        trip_generator.source_generator.write_weights(
            options.weights_outprefix + SOURCE_SUFFIX)
        trip_generator.sink_generator.write_weights(
            options.weights_outprefix + SINK_SUFFIX)
        if trip_generator.via_generator:
            trip_generator.via_generator.write_weights(
                options.weights_outprefix + VIA_SUFFIX)

    # return wether trips could be generated as requested
    return trip_generator is not None


if __name__ == "__main__":
    if not main(get_options()):
        sys.exit(1)
