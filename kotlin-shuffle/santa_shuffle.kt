package tsp

import java.io.BufferedReader
import java.io.File
import java.io.FileReader
import java.io.PrintStream
import java.util.*
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.Comparator
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.collections.HashSet
import kotlin.concurrent.thread
import kotlin.math.*

fun calcHeuristicShortPath(kdTree: KDTree, cities: List<City>): Array<City> {
    println("start")
    val connections = HashMap<City, MutableList<City>>()
    val dontconnect = HashMap<City, City>()
    val bestedge = PriorityQueue<Triple<Double, City, City>>(Comparator<Triple<Double, City, City>> {
            o1, o2 -> o1.first.compareTo(o2.first) })
    for (city in cities) {
        connections.put(city, ArrayList())
        val nneighbour = kdTree.findNearestNeighbours(city, 1).poll().second
        bestedge.add(Triple(city.calcDistance(nneighbour), city, nneighbour))
    }
    val alreadyAssigned = BooleanArray(cities.size, {false})
    val alchildrenAssigned = BooleanArray(cities.size, {false})
    var numassigned = 0
    while (numassigned != cities.size) {
        //println(numassigned)
        val (_, c1, c2) = bestedge.poll()
        if (connections[c1]!!.size == 2) {continue}
        if (connections[c2]!!.size != 2 && dontconnect[c1] != c2) {
            numassigned++
            connections[c1]!!.add(c2)
            connections[c2]!!.add(c1)
            val d1 = dontconnect[c1] ?: c1
            val d2 = dontconnect[c2] ?: c2
            dontconnect[d1] = d2
            dontconnect[d2] = d1
            alreadyAssigned[c1.num] = connections[c1]!!.size == 2
            alreadyAssigned[c2.num] = connections[c2]!!.size == 2
        }
        if (numassigned == cities.size-1) {break}
        if (connections[c1]!!.size == 2) {continue}
        val nneighbours = kdTree.findNearestNeighbours(c1, 2, alreadyAssigned = alreadyAssigned,
            alchildrenAssigned = alchildrenAssigned)
        val lneighbours = ArrayList<City>()
        while (nneighbours.isNotEmpty()) {
            lneighbours.add(nneighbours.poll().second)
            if (lneighbours.last() == dontconnect[c1]) {
                lneighbours.removeAt(lneighbours.size-1)
            }
        }
        val nneighbour = lneighbours.last()
        bestedge.offer(Triple(c1.calcDistance(nneighbour), c1, nneighbour))
    }
    val needconnecting = connections.entries.filter { it.value.size == 1 }
    println("left is ${needconnecting.size}")
    needconnecting[0].value.add(needconnecting[1].key)
    needconnecting[1].value.add(needconnecting[0].key)
    val pcities = ArrayList<City>()
    val scities = HashSet<City>()
    scities.add(cities.first())
    pcities.add(cities.first())
    while (pcities.size < cities.size) {
        val curcity = pcities.last()
        for (ncity in connections[curcity]!!) {
            if (pcities.size == 1 || pcities[pcities.size-2] != ncity) {
                pcities.add(ncity)
                if (!scities.add(ncity)) {
                    println("Fuck double assigned")
                }
                break
            }
        }
    }
    return pcities.toTypedArray()
}

fun calcPathWeight(pcities: Array<City>, calcprimepenalty:Boolean): Pair<Double, Double> {
    var weight = 0.0
    var origweight = 0.0
    for (i in 0..pcities.size-1) {
        val curcity = pcities[i]
        val nextcity = when (i == pcities.size-1) {
            true    -> pcities[0]
            false   -> pcities[i+1]
        }
        var curweight = curcity.calcDistance(nextcity)
        origweight += curweight
        if (calcprimepenalty && i % 10 == 9 && !curcity.isPrime) {
            curweight *= 1.1
        }
        weight += curweight
    }
    return Pair(weight, origweight)
}

class City(val num:Int, val x:Double, val y:Double, val isPrime:Boolean) {
    var pathindex = num
    var kdtree: KDTree? = null

    fun nextCity(pcities:Array<City>):City {
        return when (pathindex == pcities.size-1) {
            true  -> pcities.first()
            false -> pcities[pathindex+1]
        }
    }

    fun prevCity(pcities:Array<City>):City {
        return when (pathindex == 0) {
            true  -> pcities.last()
            false -> pcities[pathindex-1]
        }
    }

    fun distNextCity(pcities:Array<City>):Double {
        return calcDistance(nextCity(pcities))
    }

    fun distPrevCity(pcities:Array<City>):Double {
        return calcDistance(prevCity(pcities))
    }

    fun calcDistanceSq(ocity:City):Double{
        val dx = ocity.x-x
        val dy = ocity.y-y
        return dx * dx + dy * dy
    }

    fun calcDistance(ocity: City):Double {
        return sqrt(calcDistanceSq(ocity))
    }

    override fun toString(): String {
        return "City(num=$num, x=$x, y=$y, pathindex=$pathindex)"
    }


}

fun updateMaxOutGoing(citieswithswap: Collection<City>) {
    val updatedcities = HashSet(citieswithswap)
    val pq = PriorityQueue<KDTree>(
        Comparator<KDTree> { o1, o2 -> o2!!.depth.compareTo(o1!!.depth) })
    citieswithswap.forEach { pq.add(it.kdtree) }
    while (pq.isNotEmpty()) {
        val kdTree = pq.poll()
        val newmaxoutgoing = kdTree.calcMaxOutGoingEdge()
        if (newmaxoutgoing != kdTree.maxoutgoingedge) {
            kdTree.maxoutgoingedge = newmaxoutgoing
            if (kdTree.parent != null && updatedcities.add(kdTree.parent.city)) {
                pq.add(kdTree.parent)
            }
        }
    }
}

class ReversePlan(parent:ReversePlan?, val newPair: Pair<Int,Int>?, val pivot:Pair<Int,Int>, val edgeAtPivot: Pair<City,City>,
                  addedgain: Double, optimisticGain:Double, val removedEdge: Pair<City,City>, val pcities: Array<City>) {
    val reverses = ArrayList<Pair<Int, Int>>()
    val removedEdges = HashSet<Pair<City,City>>()
    val mustbeafter = -1

    init {
        removedEdges.add(removedEdge)
        if (parent == null) {
            if (newPair != null) {
                reverses.add(newPair)
            }
        } else {
            removedEdges.addAll(parent.removedEdges)
            if (newPair == null) {
                reverses.addAll(parent.reverses)
            } else {
                for (i in 0..mustbeafter) {
                    reverses.add(parent.reverses[i])
                }
                var insertedpair = false
                val (cs1, cs2) = newPair
                for (i in mustbeafter+1..parent.reverses.size-1) {
                    val (ps1, ps2) = parent.reverses[i]
//                    if (!insertedpair) {
//                        when {
//                            cs1 < ps1 -> {
//                                reverses.add(newPair); insertedpair = true
//                            }
//                            cs1 == ps1 && cs2 < ps2 -> {
//                                reverses.add(newPair); insertedpair = true
//                            }
//                            cs1 == ps1 && cs2 == ps2 -> {
//                                print("This should not happen, double swap")
//                                reverses.add(newPair); insertedpair = true
//                            }
//                        }
//                    }
                    reverses.add(parent.reverses[i])
                }
                if (!insertedpair) {
                    reverses.add(newPair); insertedpair = true
                }
            }
        }
    }

    val pivotedgeweight = edgeAtPivot.first.calcDistance(edgeAtPivot.second)
    val gain:Double
    init {
        fun calculateCityAtPos(pos:Int, fromDepth: Int): Int {
            var currdepth = fromDepth
            var currpos = pos
            while (currdepth >= 0) {
                val (l,r) = reverses[currdepth]
                if (currpos >= l && currpos <= r) {
                    val dl = currpos-l
                    currpos = r-dl
                }
                currdepth--
            }
            return currpos
        }

        var primeloss = 0.0
        var primegain = 0.0
        if (newPair != null) {
            var firstprime = newPair.first-1
            val rem = firstprime % 10
            firstprime += 9 - rem
            var primeconsidered = firstprime
            while (primeconsidered <= newPair.second) {
                val firstcitypositionbottom = calculateCityAtPos(primeconsidered, reverses.size-1)
                val firstcitypositionup = calculateCityAtPos(primeconsidered, reverses.size-2)
                if (!pcities[firstcitypositionbottom].isPrime) {
                    val secondcitypositionbottom = calculateCityAtPos(primeconsidered+1, reverses.size-1)
                    primeloss += pcities[firstcitypositionbottom].calcDistance(pcities[secondcitypositionbottom]) * 0.1
                }
                if (!pcities[firstcitypositionup].isPrime) {
                    val secondcitypositionup = calculateCityAtPos(primeconsidered+1, reverses.size-2)
                    primegain += pcities[firstcitypositionup].calcDistance(pcities[secondcitypositionup]) * 0.1
                }
                primeconsidered += 10
            }
        }
        gain = addedgain + (parent?.gain ?: 0.0) + primegain - primeloss
    }
    val curroptimisticgain = gain + optimisticGain + pivotedgeweight
    val hashcode:Int
    init {
        var result = pivot.hashCode()
        result = 31 * result + reverses.hashCode()
        hashcode = result
    }

    fun apply_reversal(pcities: Array<City>) {
        val citieswithswaps = HashSet<City>()

        for (reversal in reverses) {
            //println(reversal)
            val pivot = (reversal.second-reversal.first)/2
            citieswithswaps.add(pcities[reversal.first])
            citieswithswaps.add(pcities[reversal.first-1])
            citieswithswaps.add(pcities[reversal.first+1])
            citieswithswaps.add(pcities[reversal.second-1])
            citieswithswaps.add(pcities[reversal.second])
            if (reversal.second == pcities.size - 1) {
                citieswithswaps.add(pcities[0])
            } else {
                citieswithswaps.add(pcities[reversal.second+1])
            }
            for (i in 0..pivot) {
                val temp = pcities[reversal.first+i]
                pcities[reversal.first+i] = pcities[reversal.second-i]
                pcities[reversal.second-i] = temp
                pcities[reversal.first+i].pathindex = reversal.first+i
                pcities[reversal.second-i].pathindex = reversal.second-i
            }
        }
        updateMaxOutGoing(citieswithswaps)
    }

    fun calculatePositionFacingAndMustbeAfterReverseOfPos(pos:Int): Triple<Int,Boolean,Int> {
        var currpos = pos
        var facing = true
        var mustbeafter = -1
        for (i in 0..reverses.size-1) {
            val (l,r) = reverses[i]

            if (currpos < l || currpos > r) {continue}

            mustbeafter = i
            facing = !facing
            // move currpos to be the same distance from the right, that it currently is to the left
            val dl = currpos-l
            currpos = r-dl
        }
        return Triple(currpos, facing, mustbeafter)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as ReversePlan

        if (pivot != other.pivot) return false
        if (reverses != other.reverses) return false

        return true
    }

    override fun hashCode(): Int {
        return hashcode
    }
}

fun search3Opt(storedtree: KDTree, startbreakpoint: City, targetReversals: Int): ReversePlan {
    val pcities = storedtree.pcities
    val seenreversals = HashSet<ReversePlan>()
    val searchStack = PriorityQueue<Triple<Double, ReversePlan, KDTree>>(
        Comparator<Triple<Double, ReversePlan, KDTree>> { o1, o2 -> o2!!.first.compareTo(o1!!.first) })
    var optgainperreverse = 40.0
    var bestReversal = ReversePlan(null, null,
        Pair(startbreakpoint.pathindex,startbreakpoint.pathindex+1),
        Pair(startbreakpoint,startbreakpoint.nextCity(pcities)),
        0.0, (targetReversals - 1) * optgainperreverse, Pair(startbreakpoint, startbreakpoint.nextCity(pcities)),pcities)
    searchStack.offer(Triple(storedtree.maxoutgoingedge+bestReversal.curroptimisticgain, bestReversal, storedtree))
    //println()
    while (searchStack.isNotEmpty()) {
        val (optgain, reversal, kdtree) = searchStack.poll()
        if (optgain < bestReversal.gain) {
            continue
        }
        val (kdtreecitypos, facing, mustbeafter) = reversal.calculatePositionFacingAndMustbeAfterReverseOfPos(kdtree.city.pathindex)
        val othercity = when(facing) {
            true    -> kdtree.city.nextCity(pcities)
            false   -> kdtree.city.prevCity(pcities)
        }
        val thisedge = when(facing) {
            true    -> Pair(kdtree.city, othercity)
            false   -> Pair(othercity, kdtree.city)
        }
        if (!reversal.removedEdges.contains(thisedge)) {
            val removedEdgeWeights = reversal.pivotedgeweight + kdtree.city.calcDistance(othercity)
            val (oc1, oc2) = when (facing) {
                true    -> Pair(kdtreecitypos, kdtreecitypos+1)
                false   -> Pair(kdtreecitypos, kdtreecitypos+1)
            }
            val (fcity,scity) = when (facing) {
                true    -> Pair(kdtree.city, othercity)
                false   -> Pair(kdtree.city, othercity)
            }
            val reversepair = Pair(min(reversal.pivot.second, oc2), max(reversal.pivot.first, oc1))
            if (reversepair.first != reversepair.second) {
                if (reversepair.second == 0) {
                    println("removed edges ${Arrays.toString(reversal.removedEdges.toTypedArray())} rem edge size ${reversal.removedEdges.size}")
                    println("pivot")
                    println("reversals ${reversal.reverses}")
                    println("${kdtree.city}, ${othercity}")
                    println("${reversal.pivot.first}, ${reversal.pivot.second}, ${oc1}, ${oc2}, ${reversepair}")
                }
                val addEdgeWeight =
                    reversal.edgeAtPivot.first.calcDistance(fcity) + reversal.edgeAtPivot.second.calcDistance(scity)
                val (edgeAtPivotDefault, edgeAtOtherPivot) = when (oc1 < reversal.pivot.first) {
                    true -> Pair(Pair(scity, reversal.edgeAtPivot.second), Pair(fcity, reversal.edgeAtPivot.first))
                    false -> Pair(Pair(reversal.edgeAtPivot.first, fcity), Pair(reversal.edgeAtPivot.second, scity))
                }
                val newreversal1 = ReversePlan(
                    reversal, reversepair, reversal.pivot, edgeAtPivotDefault,
                    removedEdgeWeights - addEdgeWeight, (targetReversals - reversal.reverses.size) * optgainperreverse,
                    thisedge, pcities
                )
                val newreversal2 = ReversePlan(
                    reversal, reversepair, Pair(oc1, oc2), edgeAtOtherPivot,
                    removedEdgeWeights - addEdgeWeight, (targetReversals - reversal.reverses.size) * optgainperreverse,
                    thisedge, pcities
                )
                val noptgain = if (newreversal1.reverses.size == targetReversals) {
                    -1.0//newreversal1.gain
                } else {
                    newreversal1.curroptimisticgain
                }
                if (noptgain > bestReversal.gain) {
//                if (seenreversals.add(newreversal1)) {
//                    println("nr1 ${newreversal1.gain} ${newreversal1.curroptimisticgain} ${newreversal1.reverses.size}")
                    searchStack.offer(Triple(noptgain, newreversal1, storedtree))
//                }
//                if (seenreversals.add(newreversal2)) {
////                    println("nr2 ${newreversal2.gain} ${newreversal2.curroptimisticgain} ${newreversal2.reverses.size}")
//                    searchStack.offer(Triple(noptgain, newreversal2, storedtree))
//                }
                }
                if (newreversal1.gain > bestReversal.gain) {
                    bestReversal = newreversal1
                }
            }
        }
        for (subtree in listOf(kdtree.leftBranch, kdtree.rightBranch)) {
            if (subtree == null) {continue}
            val optdist1 = max(0.0,
                subtree.calcOptimisticDistance(reversal.edgeAtPivot.first) - subtree.maxoutgoingedge)
            val optdist2 = max(0.0,
                subtree.calcOptimisticDistance(reversal.edgeAtPivot.second) - subtree.maxoutgoingedge)
            val newoptimisticgain = - optdist1 - optdist2 + subtree.maxoutgoingedge + reversal.pivotedgeweight + reversal.curroptimisticgain
            if (newoptimisticgain > bestReversal.gain) {
                searchStack.offer(
                    Triple(
                        newoptimisticgain,
                        reversal, subtree
                    )
                )
            }
        }
    }
    return bestReversal
}

fun searchNearest(storedtree: KDTree, startbreakpoint: City, targetReversals: Int, closestCities: Map<City, List<City>>,
                  mustbebetterthan: Double = 0.0): ReversePlan {
    val pcities = storedtree.pcities
    val searchStack = PriorityQueue<Triple<Double, ReversePlan, KDTree>>(
        Comparator<Triple<Double, ReversePlan, KDTree>> { o1, o2 -> o2!!.first.compareTo(o1!!.first) })
    val bestreversalref = AtomicReference<ReversePlan>(ReversePlan(null, null,
        Pair(startbreakpoint.pathindex,startbreakpoint.pathindex+1),
        Pair(startbreakpoint,startbreakpoint.nextCity(pcities)),
        0.0, 0.0, Pair(startbreakpoint, startbreakpoint.nextCity(pcities)), pcities))
    val closestoinit = closestCities[startbreakpoint]!!
    val initreversal = bestreversalref.get()!!
    closestoinit.parallelStream().forEach() {
        if (it == initreversal.edgeAtPivot.second) {return@forEach}
        val curredge = Pair(it, it.nextCity(pcities))
        val removedEdgeWeights = initreversal.pivotedgeweight + it.distNextCity(pcities)
        val addedEdgeWeights = initreversal.edgeAtPivot.first.calcDistance(it) + initreversal.edgeAtPivot.second.calcDistance(it.nextCity(pcities))
        val (oc1, oc2) = Pair(it.pathindex, it.pathindex+1)
        val reversepair = Pair(min(initreversal.pivot.second, oc2), max(initreversal.pivot.first, oc1))
        val (edgeAtPivotDefault, edgeAtOtherPivot) = when (oc1 < initreversal.pivot.first) {
            true    -> Pair(Pair(curredge.second, initreversal.edgeAtPivot.second), Pair(curredge.first, initreversal.edgeAtPivot.first))
            false   -> Pair(Pair(initreversal.edgeAtPivot.first, curredge.first), Pair(initreversal.edgeAtPivot.second, curredge.second))
        }
        val secondReversal = ReversePlan(initreversal, reversepair, initreversal.pivot, edgeAtPivotDefault,
            removedEdgeWeights - addedEdgeWeights, 0.0, curredge, pcities)
        synchronized(bestreversalref) {
            if (secondReversal.gain > bestreversalref.get()!!.gain) {
                bestreversalref.set(secondReversal)
            }
        }
        if (targetReversals == 1) {return@forEach}
        val stack = ArrayList<ReversePlan>()
        stack.add(secondReversal)
        while (stack.isNotEmpty()) {
            val currreversal = stack.removeAt(stack.size-1)
            val currclosest = closestCities[currreversal.edgeAtPivot.first]!!.reversed()
            for (city in currclosest) {
                val (citypos, facing, mustbeafter) = currreversal.calculatePositionFacingAndMustbeAfterReverseOfPos(city.pathindex)
                val othercity = when (facing) {
                    true -> city.nextCity(pcities)
                    false -> city.prevCity(pcities)
                }
                val thisedge = when (facing) {
                    true -> Pair(city, othercity)
                    false -> Pair(othercity, city)
                }
                val (oc1, oc2) = Pair(citypos, citypos + 1)
                val (fcity, scity) = Pair(city, othercity)
                val reversepair = Pair(min(currreversal.pivot.second, oc2), max(currreversal.pivot.first, oc1))
                if (!currreversal.removedEdges.contains(thisedge) && reversepair.first != reversepair.second) {
                    val removedEdgeWeights = currreversal.pivotedgeweight + city.calcDistance(othercity)
                    val addEdgeWeight =
                        currreversal.edgeAtPivot.first.calcDistance(fcity) + currreversal.edgeAtPivot.second.calcDistance(scity)
                    val (edgeAtPivotDefault, edgeAtOtherPivot) = when (oc1 < currreversal.pivot.first) {
                        true    -> Pair(Pair(scity, currreversal.edgeAtPivot.second), Pair(fcity, currreversal.edgeAtPivot.first))
                        false   -> Pair(Pair(currreversal.edgeAtPivot.first, fcity), Pair(currreversal.edgeAtPivot.second, scity))
                    }
                    val newreversal1 = ReversePlan(
                        currreversal, reversepair, currreversal.pivot, edgeAtPivotDefault,
                        removedEdgeWeights - addEdgeWeight, 0.0,
                        thisedge, pcities
                    )
                    val newreversal2 = ReversePlan(
                        currreversal, reversepair, Pair(oc1, oc2), edgeAtOtherPivot,
                        removedEdgeWeights - addEdgeWeight, 0.0,
                        thisedge, pcities
                    )
                    synchronized(bestreversalref) {
                        if (bestreversalref.get().gain < newreversal1.gain ||
                            bestreversalref.get() == initreversal &&
                            newreversal1.gain >= mustbebetterthan) {
                            bestreversalref.set(newreversal1)
                        }
                    }
                    if (newreversal1.reverses.size < targetReversals) {
                        stack.add(newreversal1)
                        stack.add(newreversal2)
                    }
                }
            }
        }
    }
    return bestreversalref.get()!!
}

class KDTree {
    val parent:KDTree?
    val city:City
    val depth:Int
    val leftBranch:KDTree?
    val rightBranch:KDTree?
    val gx:Double
    val gy:Double
    val lx:Double
    val ly:Double
    val pcities:Array<City>
    var maxoutgoingedge:Double

    constructor(cities:List<City>, sortonX:Boolean, pcities:Array<City>, d:Int, p: KDTree? = null) {
        parent = p
        val sortedcities = ArrayList<City>(cities)
        depth = d
        this.pcities = pcities
        when (sortonX) {
            true -> sortedcities.sortBy { it.x }
            false -> sortedcities.sortBy { it.y }
        }
        gx = sortedcities.maxBy { it.x }!!.x
        gy = sortedcities.maxBy { it.y }!!.y
        lx = sortedcities.minBy { it.x }!!.x
        ly = sortedcities.minBy { it.y }!!.y
        val medindex = sortedcities.size/2
        city = sortedcities[medindex]
        city.kdtree = this
        val lessercities = sortedcities.subList(0, medindex)
        val greatercities = sortedcities.subList(medindex+1, sortedcities.size)
        fun createBranch(subcities:List<City>):KDTree? {
            if (subcities.isEmpty()) {
                return null;
            } else {
                return KDTree(subcities, !sortonX, pcities, depth+1)
            }
        }
        leftBranch = createBranch(lessercities)
        rightBranch = createBranch(greatercities)
        maxoutgoingedge = calcMaxOutGoingEdge()
    }



    fun calcMaxOutGoingEdge():Double {
        val mgo = max(leftBranch?.maxoutgoingedge ?: 0.0,rightBranch?.maxoutgoingedge ?: 0.0)
        return max(mgo, max(city.distPrevCity(pcities), city.distNextCity(pcities)))
    }

    fun updateMaxOutGoingRecursive() {
        leftBranch?.updateMaxOutGoingRecursive()
        rightBranch?.updateMaxOutGoingRecursive()
        maxoutgoingedge = max(leftBranch?.maxoutgoingedge ?: 0.0, rightBranch?.maxoutgoingedge ?: 0.0)
        maxoutgoingedge = max(maxoutgoingedge, max(city.distPrevCity(pcities), city.distNextCity(pcities)))
//        println("${city.distPrevCity(pcities)}, ${city.distNextCity(pcities)}")
//        println(maxoutgoingedge)
    }

    fun calcOptimisticDistanceSQ(ocity:City):Double {
        val cycoord:Double
        val cxcoord:Double
        if (ocity.y > gy) {
            cycoord = gy
        } else if (ocity.y < ly) {
            cycoord = ly
        } else {
            cycoord = ocity.y
        }
        if (ocity.x > gx) {
            cxcoord = gx
        } else if (ocity.x < lx) {
            cxcoord = lx
        } else {
            cxcoord = ocity.x
        }
        val dy = cycoord - ocity.y
        val dx = cxcoord - ocity.x
        return dx * dx + dy * dy
    }

    fun calcOptimisticDistance(ocity:City):Double {
        return sqrt(calcOptimisticDistanceSQ(ocity))
    }

    /**
     * This can possibly be improved based upon taking into account the relative position of incity, outcity, and the bounding box.
     */
    fun calcOptimisticSwap(incity:Int, outcity:Int, currentCost:Double): Double {
        val optimisticdistincity = calcOptimisticDistance(pcities[incity])
        val optimisticdistoutcity = max(0.0,calcOptimisticDistance(pcities[outcity]) - maxoutgoingedge)
        return currentCost + optimisticdistincity + optimisticdistoutcity - maxoutgoingedge
    }

    fun findNearestNeighbours(tofind:City, n:Int,
                              currclosest: PriorityQueue<Pair<Double, City>> =
                                  PriorityQueue(Comparator<Pair<Double, City>> { o1, o2 -> o2!!.first.compareTo(o1!!.first) }),
                              alreadyAssigned: BooleanArray? = null, alchildrenAssigned: BooleanArray? = null,
                              radius: Double = Double.POSITIVE_INFINITY):
            PriorityQueue<Pair<Double, City>> {
        if (n == 0) {return currclosest}
        fun exploreBranch(branch:KDTree?, dist: Double) {
            if (branch != null && dist < radius) {
                val branchalreadyassigned = alchildrenAssigned?.get(branch.city.num) ?: false
                if (!branchalreadyassigned && (currclosest.size < n || dist < currclosest.peek().first)) {
                    branch.findNearestNeighbours(tofind, n, currclosest, alreadyAssigned, alchildrenAssigned, radius)
                }
            }
        }
        fun exploreBranches() {
            val ldist = leftBranch?.calcOptimisticDistanceSQ(tofind) ?: Double.POSITIVE_INFINITY
            val rdist = rightBranch?.calcOptimisticDistanceSQ(tofind) ?: Double.POSITIVE_INFINITY
            if (ldist < rdist) {
                exploreBranch(leftBranch, ldist)
                exploreBranch(rightBranch, rdist)
            } else {
                exploreBranch(rightBranch, rdist)
                exploreBranch(leftBranch, ldist)
            }
        }
        val isAssigned = alreadyAssigned?.get(city.num) ?: false
        if (tofind != city && !isAssigned && tofind.calcDistance(city) < radius) {
            val sqdist = tofind.calcDistanceSq(city)
            currclosest.add(Pair(sqdist, city))
            //println(Arrays.toString(currclosest.toArray()))
            while (currclosest.size > n)  {currclosest.poll()}
            //println(Arrays.toString(currclosest.toArray()))
        }
        exploreBranches()
        if (isAssigned && alchildrenAssigned != null) {
            val alassignedleft = leftBranch == null || alchildrenAssigned[leftBranch.city.num]
            val alassignedright = rightBranch == null || alchildrenAssigned[rightBranch.city.num]
            if (alassignedleft && alassignedright) {
                alchildrenAssigned[city.num] = true
            }
        }
        return currclosest
    }

    fun findNearestTen(tofind:City, n:Int, currclosest: PriorityQueue<Pair<Double, City>> =
        PriorityQueue(Comparator<Pair<Double, City>> { o1, o2 -> o2!!.first.compareTo(o1!!.first) }),
                       alreadyAssigned: BooleanArray? = null, alchildrenAssigned: BooleanArray? = null):
            PriorityQueue<Pair<Double, City>> {
        fun exploreBranch(branch:KDTree?, dist: Double) {
            if (branch != null) {
                val branchalreadyassigned = alchildrenAssigned?.get(branch.city.num) ?: false
                if (!branchalreadyassigned && (currclosest.size < n || dist < currclosest.peek().first)) {
                    branch.findNearestTen(tofind, n, currclosest, alreadyAssigned, alchildrenAssigned)
                }
            }
        }
        fun exploreBranches() {
            val ldist = leftBranch?.calcOptimisticDistanceSQ(tofind) ?: Double.POSITIVE_INFINITY
            val rdist = rightBranch?.calcOptimisticDistanceSQ(tofind) ?: Double.POSITIVE_INFINITY
            if (ldist < rdist) {
                exploreBranch(leftBranch, ldist)
                exploreBranch(rightBranch, rdist)
            } else {
                exploreBranch(rightBranch, rdist)
                exploreBranch(leftBranch, ldist)
            }
        }
        val isAssigned = alreadyAssigned?.get(city.num) ?: false
        if (tofind != city && !isAssigned && !city.isPrime) {
            val sqdist = tofind.calcDistanceSq(city)
            currclosest.add(Pair(sqdist, city))
            //println(Arrays.toString(currclosest.toArray()))
            while (currclosest.size > n)  {currclosest.poll()}
            //println(Arrays.toString(currclosest.toArray()))
        }
        exploreBranches()
        if (isAssigned && alchildrenAssigned != null) {
            val alassignedleft = leftBranch == null || alchildrenAssigned[leftBranch.city.num]
            val alassignedright = rightBranch == null || alchildrenAssigned[rightBranch.city.num]
            if (alassignedleft && alassignedright) {
                alchildrenAssigned[city.num] = true
            }
        }
        return currclosest
    }

    fun findNearestNeighboursFacing(tofind:City, toWard:City, n:Int,
                                    currclosest: PriorityQueue<Pair<Double, City>> =
                                        PriorityQueue(Comparator<Pair<Double, City>> { o1, o2 -> o2!!.first.compareTo(o1!!.first) }),
                                    alreadyAssigned: BooleanArray? = null, alchildrenAssigned: BooleanArray? = null):
            PriorityQueue<Pair<Double, City>> {
        val above:Boolean = toWard.y > tofind.y
        val toright:Boolean = toWard.x > tofind.x
        fun exploreBranch(branch:KDTree?, dist: Double) {
            if (branch != null) {
//                when {
//                    above && branch.gy < tofind.y   -> return
//                    !above && branch.ly > tofind.y   -> return
//                    toright && branch.gx < tofind.x   -> return
//                    !toright && branch.lx > tofind.x   -> return
//                }
                if (calcOptimisticDistance(toWard) > toWard.calcDistance(tofind)){
                    return
                }
                val branchalreadyassigned = alchildrenAssigned?.get(branch.city.num) ?: false
                if (!branchalreadyassigned && (currclosest.size < n || dist < currclosest.peek().first)) {
                    branch.findNearestNeighboursFacing(tofind, toWard, n, currclosest, alreadyAssigned, alchildrenAssigned)
                }
            }
        }
        fun exploreBranches() {
            val ldist = leftBranch?.calcOptimisticDistanceSQ(tofind) ?: Double.POSITIVE_INFINITY
            val rdist = rightBranch?.calcOptimisticDistanceSQ(tofind) ?: Double.POSITIVE_INFINITY
            if (ldist < rdist) {
                exploreBranch(leftBranch, ldist)
                exploreBranch(rightBranch, rdist)
            } else {
                exploreBranch(rightBranch, rdist)
                exploreBranch(leftBranch, ldist)
            }
        }
        val isAssigned = alreadyAssigned?.get(city.num) ?: false
        if (tofind != city && !isAssigned) {
            if (toWard.calcDistance(city) < toWard.calcDistance(tofind)) {
                val sqdist = tofind.calcDistanceSq(city)
                currclosest.add(Pair(sqdist, city))
//            println(Arrays.toString(currclosest.toArray()))
                while (currclosest.size > n) {
                    currclosest.poll()
                }
//            println(Arrays.toString(currclosest.toArray()))
            }
        }
        exploreBranches()
        if (isAssigned && alchildrenAssigned != null) {
            val alassignedleft = leftBranch == null || alchildrenAssigned[leftBranch.city.num]
            val alassignedright = rightBranch == null || alchildrenAssigned[rightBranch.city.num]
            if (alassignedleft && alassignedright) {
                alchildrenAssigned[city.num] = true
            }
        }
        return currclosest
    }
}

fun removeLargestEdge(kdTree: KDTree, pcities: Array<City>, numedges: Int = 100, minedgesize: Double = 100.0) {
    val allcities = HashSet(pcities.asList())
    if (allcities.size < pcities.size) {
        println("fuck not all assigned")
    }
    val edgepq = PriorityQueue<Triple<City, City, Double>>(Comparator<Triple<City,City,Double>>{
            t1, t2 -> t1.third.compareTo(t2.third)
    })
    edgepq.add(Triple(pcities.last(), pcities.first(), pcities.first().calcDistance(pcities.last())))
    //var largestedge = Triple(pcities.last(), pcities.first(), pcities.first().calcDistance(pcities.last()))
    for (i in 0..pcities.size-2) {
        val dist = pcities[i].calcDistance(pcities[i+1])
        edgepq.offer(Triple(pcities[i], pcities[i+1], dist))
        while (edgepq.isNotEmpty() && (edgepq.size > numedges || edgepq.peek().third < minedgesize)) {
            edgepq.poll()
        }
    }
    println("largest edge is ${edgepq.peek()}")
    val edgeorder = ArrayList<Triple<City,City,Double>>()
    while (edgepq.isNotEmpty()) {
        edgeorder.add(edgepq.poll())
    }
    edgeorder.reverse()
    for (largestedge in edgeorder) {
        if ((largestedge.first.pathindex + 1)%pcities.size != largestedge.second.pathindex) {
            continue
        }
        val newrun = ArrayList<City>()
        newrun.add(largestedge.first)
        while (newrun.last() != largestedge.second) {
            val nextcity = kdTree.findNearestNeighboursFacing(newrun.last(), largestedge.second, 1).poll().second
            newrun.add(nextcity)
//        println(newrun.last())
//        println(nextcity.calcDistance(largestedge.second))
        }
        val removedcities = HashSet<City>(newrun)
        val neworder = ArrayList<City>()
        for (i in 0..pcities.size-1) {
            if (pcities[i] == newrun[0]) {
                neworder.addAll(newrun)
            }
            if (!removedcities.contains(pcities[i])) {
                neworder.add(pcities[i])
            }
        }
        for (i in 0..neworder.size-1) {
            kdTree.pcities[i] = neworder[i]
            neworder[i].pathindex = i
        }
    }
    //updateMaxOutGoing(pcities.asList())
    kdTree.updateMaxOutGoingRecursive()
}



fun startsolver(indexoffset:Int, indexstep:Int, lockobject:Any) {
    val dropboxdirstring = """D:\Dropbox\kick-package\tours\"""
    //System.loadLibrary("jniortools")
    val reader = BufferedReader(FileReader("cities.csv"))
    val closestcities = Collections.synchronizedMap(HashMap<City, List<City>>())
    reader.readLine()
    val primes = calcprimes(1000000)
    val cities = ArrayList<City>()
    while (reader.ready()) {
        val split = reader.readLine().trim().split(",")
        cities.add(City(split[0].toInt(), split[1].toDouble(), split[2].toDouble(), primes.contains(split[0].toInt())))
    }
    val kdtree = KDTree(cities, true, cities.toTypedArray(), 0)
    cities.parallelStream().forEach() {
        val nneighbours = kdtree.findNearestNeighbours(it, 1, radius = 50.0)
        val neighbourlist = ArrayList<City>()
        while (nneighbours.isNotEmpty()) {
            neighbourlist.add(nneighbours.poll().second)
        }
        neighbourlist.reverse()
        closestcities.put(it, neighbourlist)
    }
    println("neighbours done")

    fun load(fname:String) {
        val initpath = BufferedReader(FileReader(fname))
        initpath.readLine()
        kdtree.pcities[0] = cities[0]
        var index = 0
        while (initpath.ready()) {
            val cnum = initpath.readLine().toInt()
            if (cnum != 0) {
                kdtree.pcities[index] = cities[cnum]
                cities[cnum].pathindex = index
            }
            index++
        }
        kdtree.updateMaxOutGoingRecursive()
    }
    fun findSmallestPath():Pair<String,Double> {
        val tourdirs = File(dropboxdirstring)
        val minfile = tourdirs.list().minBy {
            if (it.length > 16) {
                return@minBy it.subSequence(5, 17).toString().toDouble()!!
            }
            return@minBy Double.POSITIVE_INFINITY
        }
        println(minfile)
        val weight = minfile!!.subSequence(5, 17).toString().toDouble()
        println(weight)
        return Pair(dropboxdirstring + minfile, weight)
    }
    load(findSmallestPath().first)
    findSmallestPath()
    println("weight = ${calcPathWeight(kdtree.pcities, true)}")
    println("maxoutgoing = ${kdtree.maxoutgoingedge}")
    //kdtree.updateMaxOutGoingRecursive()
    val totaledgesnonpenalty = kdtree.pcities.filter {it.isPrime || it.pathindex % 10 != 9}.sumByDouble { it.distNextCity(kdtree.pcities) }
    val nonpenaltyedges = kdtree.pcities.filter {it.isPrime || it.pathindex % 10 != 9}.size
    val totaledgespenalty = kdtree.pcities.filter {!it.isPrime && it.pathindex % 10 == 9}.sumByDouble { it.distNextCity(kdtree.pcities) }
    val penaltyedges = kdtree.pcities.filter {!it.isPrime && it.pathindex % 10 == 9}.size
    println("totalnonpen $totaledgesnonpenalty nonpen $nonpenaltyedges totalpenedge $totaledgespenalty penedge $penaltyedges")
    println("avenonpen ${totaledgesnonpenalty/nonpenaltyedges} avepenedge ${totaledgespenalty/penaltyedges}")
    println(kdtree.pcities.filter {!it.isPrime && it.pathindex % 10 == 9}.maxBy { it.distNextCity(kdtree.pcities) }!!.distNextCity(kdtree.pcities))
    var primesonten = kdtree.pcities.count {it.isPrime && it.pathindex % 10 == 9}
    val worstpenout = kdtree.pcities.filter {!it.isPrime && it.pathindex % 10 == 9}.maxBy { it.distNextCity(kdtree.pcities) }!!
    for (i in worstpenout.pathindex-20..worstpenout.pathindex+20) {
        println("${kdtree.pcities[i]} dist ${kdtree.pcities[i].distNextCity(kdtree.pcities)}")
    }
    println("$primesonten vs ${kdtree.pcities.count {it.isPrime}}")
    var run = 0
    var lastweight = calcPathWeight(kdtree.pcities, true)
    var waschanged = true
    var targetReversals = 1
    val DYNAMICKICK = 3
    val DYNAMIC1 = 1
    val DYNAMIC2 = 2
    val REVERSAL = 0
    var usedynamic = DYNAMIC1
    val stride = 100
    val maxfringe = 1000
    val stuff = ArrayList<Int>()
    val random = Random(0)
//    movePrimeToTen(kdtree)
    if (usedynamic == DYNAMIC1) {
        stuff.addAll((0..(cities.size / stride) + 1).shuffled().asSequence())
    } else {
        stuff.addAll((0..500).shuffled().asSequence())
    }
    var currweight = calcPathWeight(kdtree.pcities, true).first
    while (waschanged) {
        val runmod = ((run+19)*13)%stride
        var topi = indexoffset
        while (topi < stuff.size) {
            val i = stuff[topi]
            topi+=indexstep
            val start = max(1, i * stride + runmod)
            val end = min(cities.size - 1, start + stride)
            if (start >= end) {
                continue
            }
            println("start:$start, end:$end")
            val prevweight = currweight
            currweight = DynamicPathSearch(kdtree.pcities, start, end, maxfringe, currweight)
            val (toursfile, tourweight) = findSmallestPath()
            val diff = tourweight - currweight
            if (diff < -0.0001) {
                synchronized(lockobject) {
                    println("take tour ")
                    load(toursfile)
                    if (currweight < prevweight) {
                        topi-=indexstep
                    }
                    currweight = tourweight
                }
            } else if (diff > 0.0001) {
                synchronized(lockobject) {
                    println("updated")
                    outputtofile(dropboxdirstring, kdtree.pcities)
                }
            } else {
                //println("statusquo")
            }
        }
        run++
    }
    val ints = ArrayList<Int>()
    val pcities = kdtree.pcities
    ints.addAll(0..cities.size-2)
    ints.parallelStream().forEach() {
        val i = it
        val j = i+1
        for (k in i+1..cities.size-1) {
            val l = (k + 1) % cities.size
            val removed = pcities[i].distNextCity(pcities) + pcities[k].distNextCity(pcities)
            val added = pcities[i].calcDistance(pcities[k]) + pcities[j].calcDistance(pcities[l])
            if (removed > added) {
                println("$i,$j,$k,$l, removede:$removed, $added, ${pcities[i]}, ${pcities[j]}, ${pcities[k]}, ${pcities[l]}")
            }
        }
    }
}

fun main(args: Array<String>) {
    val step = 12
    val lockobject = ArrayList<Any>()
    for (i in 0..step-1) {
        thread { startsolver(i, step, lockobject)}
    }
}

fun calcprimes(topprime: Int): Set<Int> {
    val primes = HashSet<Int>()
    val isPrime = BooleanArray(topprime+1, {true})
    val intsqrt = (sqrt(topprime.toDouble()) + 1).toInt()
    for (i in 2..topprime) {
        if (isPrime[i]) {
            primes.add(i)
            if (i > intsqrt) {continue}
            val startval = i * i
            for (j in startval..topprime step i) {
                isPrime[j] = false
            }
        }
    }
    return primes
}


fun outputtofile(dir :String, path:Array<City>) {
    val pstream = PrintStream(dir + "tour-${calcPathWeight(path, true).first}-kha-multithread.csv")
    pstream.println("Path")
    path.forEach { pstream.println(it.num) }
    pstream.println("0")
    pstream.close()
}

fun DynamicPathSearch(pcities:Array<City>, from: Int, to:Int, maxfringe: Int = 100, currweight: Double):Double {
    val only = 10
    val extrarimepenalty = 1
    val rightmostmayhavepenalty = (to-1)%10 == 9
    val closestCities = HashMap<City, List<Pair<Double, City>>>()
    var origcost = pcities[from-1].calcDistance(pcities[from])
    if ((from-1) % 10 == 9 && !pcities[from-1].isPrime) {
        origcost *= 1.1
    }
    var primesleft = 0
    for (i in from..to-1) {
        if (i %10 == 9) {primesleft++}
        when(i%10 == 9 && !pcities[i].isPrime) {
            true    -> origcost += pcities[i].distNextCity(pcities) * 1.1
            false   -> origcost += pcities[i].distNextCity(pcities)
        }
    }
    val cities = ArrayList(pcities.asList().subList(from, to))
    cities.addAll(listOf(pcities[from-1], pcities[to]))
    val firstSecondCity = HashMap<City, Pair<Int,Int>>()
    for (city in cities) {
        val neighbours = ArrayList(cities.filter {
            it != city
        }.map { Pair(it.calcDistance(city), it)})
        neighbours.sortBy { it.first }
        closestCities.put(city, neighbours)
        firstSecondCity.put(city, Pair(0,0))
    }
    val lo = Leftover(pcities[from-1], pcities[to],
        HashSet(pcities.asList().subList(from, to)), closestCities, 0,
        primesleft, (from-1) % 10 == 9 && !pcities[from-1].isPrime,
        HashMap(firstSecondCity), rightmostmayhavepenalty)
//    val timeorigheur = System.currentTimeMillis()
//    lo.heurmin()
//    println("orig took ${System.currentTimeMillis()-timeorigheur}")
//    val timenewheur = System.currentTimeMillis()
//    lo.heuristicmin3()
//    println("new took ${System.currentTimeMillis()-timenewheur}")
//    println("current $origcost vs lowerbound ${lo.heurmin()} vs origlower bound ${lo.origlowerbound}" +
//            "")
    var pq = TreeSet<Pair<Leftover, cityLink>>(Comparator<Pair<Leftover, cityLink>> {
            p1, p2 ->
        when (p2.second.cost.compareTo(p1.second.cost)) {
            0   -> p1.first.leftovernum.compareTo(p2.first.leftovernum)
            else-> p1.second.cost.compareTo(p2.second.cost)
        }
    })
    pq.add(Pair(lo, cityLink(null, pcities[from-1], 0.0, lo.heurmin())))
    for (i in from..to-2) {
        //println("$i ${pq.first().first.citiesLeft.size}")
        val newleftovermap = HashMap<Leftover, Pair<Leftover, cityLink>>()
        val newpq = TreeSet<Pair<Leftover, cityLink>>(Comparator<Pair<Leftover, cityLink>> {
                p1, p2 ->
            val firstcompare =
                (p1.first.heurmin() + p1.second.cost).compareTo(
                    p2.first.heurmin() + p2.second.cost
                )
            when (firstcompare) {
                0   -> p1.first.leftovernum.compareTo(p2.first.leftovernum)
                else-> firstcompare
            }
        })
        val needsprime = i%10 == 9
        if (i % 10 == 9) {
            primesleft--;
        }
        for (prevassign in pq) {
            val citiestoconsider = ArrayList<City>()
            for (city in closestCities[prevassign.first.leftmost]!!) {
                if (prevassign.first.citiesLeft.contains(city.second)) {
                    citiestoconsider.add(city.second)
                    if (citiestoconsider.size >= only) {
                        break
                    }
                }
            }
//            citiestoconsider.addAll(prevassign.first.citiesLeft)
            for (city in citiestoconsider) {
                var edgecost = prevassign.first.leftmost.calcDistance(city)
                if ((i-1)%10 == 9 && !prevassign.first.leftmost.isPrime) {
                    edgecost *= 1.1
                }
                val currcost = prevassign.second.cost + edgecost
                val newleftover = Leftover(city, prevassign.first.rightmost,
                    prevassign.first.citiesLeft.minus(city),
                    closestCities, newleftovermap.size, primesleft,
                    needsprime && !city.isPrime, prevassign.first.firstSecondCity.toMutableMap(),
                    rightmostmayhavepenalty)
                val currassign = newleftovermap.get(newleftover)
                if (currassign != null) {
                    if (currassign.second.cost <= currcost) {continue}
                    newpq.remove(currassign)
                    val newassign = Pair(currassign.first, cityLink(prevassign.second, city,
                        currcost, currcost + currassign.first.heurmin()))
                    newpq.add(newassign)
                    newleftovermap.put(newassign.first, newassign)
                } else {
                    if (newleftover.heurmin() + currcost >= origcost) {continue}
                    val newassign = Pair(newleftover, cityLink(prevassign.second, city, currcost,
                        newleftover.heurmin() + currcost + newleftover.primepenalty*extrarimepenalty))
                    if (newpq.size >= maxfringe && newpq.last().second.lowerbound < newassign.second.lowerbound) {continue}
                    newpq.add(newassign)
                    newleftovermap.put(newassign.first, newassign)
                    while (newpq.size > maxfringe) {
                        val toremove = newpq.pollLast()
                        newleftovermap.remove(toremove.first)
                    }
                }
            }
        }
        pq = newpq
    }
    if (pq.isEmpty()) {
        //println("No result")
    } else {
        val bestone = pq.first()
        val lastcity = bestone.first.citiesLeft.iterator().next()
        var newcost = bestone.second.cost
        when ((to-2)% 10 == 9 && !bestone.second.city.isPrime) {
            true    -> newcost += bestone.second.city.calcDistance(lastcity) * 1.1
            false   -> newcost += bestone.second.city.calcDistance(lastcity)
        }
        when ((to-1)% 10 == 9 && !lastcity.isPrime) {
            true    -> newcost += pcities[to].calcDistance(lastcity) * 1.1
            false   -> newcost += pcities[to].calcDistance(lastcity)
        }
        if (newcost >= origcost) {return currweight}
        //println("Cities Left ${bestone.first.citiesLeft.size}")
        println("${newcost} vs $origcost")
        val initialpathcost = calcPathWeight(pcities, true)
        pcities[to-1] = lastcity
        bestone.first.citiesLeft.iterator().next().pathindex = to-1
        var insertpoint = to-2
        var currlink = bestone.second
        while (true) {
            pcities[insertpoint] = currlink.city
            currlink.city.pathindex = insertpoint
            if (currlink.parent == null) {
                break
            } else {
                insertpoint--
                currlink = currlink.parent!!
            }
        }
        val newpathcost = calcPathWeight(pcities, true)
        println("$initialpathcost vs $newpathcost, from =$from, to =$to")
        return newpathcost.first
    }
    return currweight
}

data class cityLink(val parent: cityLink?, val city:City, val cost: Double, val lowerbound: Double)

class Leftover(val leftmost: City, val rightmost: City, val citiesLeft: Set<City>,
               val closestCities: Map<City, List<Pair<Double, City>>>, val leftovernum:Int,
               val primesleft: Int, val leftmostHasPenalty: Boolean, val firstSecondCity: MutableMap<City, Pair<Int,Int>>,
               val rightmostmayhavepenalty: Boolean) {
    var heuristicminimum: Double = Double.POSITIVE_INFINITY
    var origlowerbound: Double = Double.POSITIVE_INFINITY
    var primepenalty: Double = Double.POSITIVE_INFINITY
    fun heurmin(): Double {
        if (heuristicminimum != Double.POSITIVE_INFINITY) {
            return heuristicminimum
        }
//        if (true) {
//            heuristicminimum = heuristicmin3()
//            return heuristicminimum
//        }
        val primesavailable = citiesLeft.count {it.isPrime}
        var primesnotcovered = max(primesleft-primesavailable, 0)
        if (primesavailable == 0 && rightmostmayhavepenalty) {
            primesnotcovered -= 1
        }
        var heurval = 0.0
        val cityusingcity = HashMap<City, MutableList<City>>()
        citiesLeft.forEach {cityusingcity.put(it, ArrayList())}
        cityusingcity.put(leftmost, ArrayList())
        cityusingcity.put(rightmost, ArrayList())
        var minoutgoing = PriorityQueue<Double>(Comparator<Double> {d1, d2 -> d2.compareTo(d1)})
        for (city in citiesLeft) {
            var taken = 0
            val neighbours = closestCities[city]!!
            var (firstindex, secondindex) = firstSecondCity[city]!!
            var currindex = firstindex
            var nearestwasleftmostorrightmost = false
            while (true) {
                val neighbour = neighbours[currindex]
                if (neighbour.second == leftmost || neighbour.second == rightmost ||
                    citiesLeft.contains(neighbour.second)) {
                    taken++
                    heurval += neighbour.first
                    cityusingcity[neighbour.second]!!.add(city)
                    if (taken == 1) {
                        firstindex = currindex
                        currindex = max(currindex, secondindex - 1)
                        if (neighbour.second == leftmost || (neighbour.second == rightmost && primesavailable == 0)) {
                            nearestwasleftmostorrightmost = true
                        } else {
                            if (!city.isPrime) {
                                minoutgoing.offer(neighbour.first)
                                while (minoutgoing.size > primesnotcovered) {
                                    minoutgoing.poll()
                                }
                            }
                        }
                    } else if (taken == 2) {
                        if (nearestwasleftmostorrightmost) {
                            if (!city.isPrime) {
                                minoutgoing.offer(neighbour.first)
                                while (minoutgoing.size > primesnotcovered) {
                                    minoutgoing.poll()
                                }
                            }
                        }
                        secondindex = currindex
                        break
                    }
                }
                currindex++
            }
            firstSecondCity.put(city, Pair(firstindex, secondindex))
        }
        var leftmostout = 0.0
        var rightmostout = 0.0
        for (city in listOf(leftmost, rightmost)) {
            var (firstindex, secondindex) = firstSecondCity[city]!!
            var currindex = firstindex
            val neighbours = closestCities[city]!!
            while (true) {
                val neighbour = neighbours[currindex]
                if (citiesLeft.contains(neighbour.second)) {
                    firstindex = currindex
                    heurval += neighbour.first
                    cityusingcity[neighbour.second]!!.add(city)
                    if (city == leftmost) {
                        leftmostout = neighbour.first
                    } else {
                        rightmostout = neighbour.first
                    }
                    if (leftmostHasPenalty && city == leftmost) {
                        minoutgoing.offer(neighbour.first)
                    }
                    break
                }
                currindex++
            }
            firstSecondCity.put(city, Pair(firstindex, firstindex))
        }
        heuristicminimum = (heurval/2)
        primepenalty = 0.0
        if (leftmostHasPenalty) {
            primepenalty += leftmostout * 0.1

        }
        if (rightmostmayhavepenalty && primesavailable == 0) {
            primepenalty += rightmostout * 0.1
        }
        for (d in minoutgoing) {
            primepenalty += d * 0.1
        }
        heuristicminimum += primepenalty
        val pushtothirdcosts = HashMap<City, MutableList<Double>>()
        for (entry in cityusingcity.entries) {
            val maxedgesto = when (entry.key == leftmost || entry.key == rightmost) {
                true    -> 1
                false   -> 2
            }
            if (entry.value.size > maxedgesto) {pushtothirdcosts.put(entry.key, ArrayList())}
        }
        for (city in citiesLeft) {
            var usethird = false
            var replacefirst = false
            var replacesecond = false
            val closecities = closestCities[city]!!
            val (firstcityind, secondcityind) = firstSecondCity[city]!!
            val maxcitiesfirst = when (closecities[firstcityind].second == leftmost ||
                    closecities[firstcityind].second == rightmost) {
                true    -> 1
                false   -> 2
            }
            val maxcitiessecond = when (closecities[secondcityind].second == leftmost ||
                    closecities[secondcityind].second == rightmost) {
                true    -> 1
                false   -> 2
            }
            replacefirst = maxcitiesfirst < cityusingcity[closecities[firstcityind].second]!!.size
            replacesecond = maxcitiessecond < cityusingcity[closecities[secondcityind].second]!!.size
            usethird = replacefirst || replacesecond
            if (usethird) {
                var currindex = secondcityind+1
                val neighbours = closecities
                while (true) {
                    val neighbour = neighbours[currindex]
                    if (neighbour.second == leftmost || neighbour.second == rightmost ||
                        citiesLeft.contains(neighbour.second)) {
                        if (replacefirst) {pushtothirdcosts[closecities[firstcityind].second]!!.add(
                            closecities[currindex].first - closecities[firstcityind].first
                        )}
                        if (replacesecond) {pushtothirdcosts[closecities[secondcityind].second]!!.add(
                            closecities[currindex].first - closecities[secondcityind].first
                        )}
                        break
                    }
                    currindex++
                }
            }
        }
        for (city in listOf(leftmost, rightmost)) {
            var replacefirst = false
            val closecities = closestCities[city]!!
            val (firstcityind, secondcityind) = firstSecondCity[city]!!
            val maxcitiesfirst = when (closecities[firstcityind].second == leftmost ||
                    closecities[firstcityind].second == rightmost) {
                true    -> 1
                false   -> 2
            }
            replacefirst = maxcitiesfirst < cityusingcity[closecities[firstcityind].second]!!.size
            if (replacefirst) {
                var currindex = secondcityind + 1
                val neighbours = closecities
                while (true) {
                    val neighbour = neighbours[currindex]
                    if (citiesLeft.contains(neighbour.second)) {
                        pushtothirdcosts[closecities[firstcityind].second]!!.add(
                            closecities[currindex].first - closecities[firstcityind].first)
                        break
                    }
                    currindex++
                }
            }
        }
        origlowerbound = heuristicminimum
        for (entry in pushtothirdcosts.entries) {
            entry.value.sort()
//            println(entry.value)
            val maxedgesto = when (entry.key == leftmost || entry.key == rightmost) {
                true    -> 1
                false   -> 2
            }
//            println("sublist ${entry.value.subList(0, entry.value.size-maxedgesto)} $maxedgesto")
            for (v in entry.value.subList(0, entry.value.size-maxedgesto)) {
                heuristicminimum += v/2
            }
        }
        origlowerbound = heuristicminimum

        return heuristicminimum
    }


    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Leftover

        if (leftmost != other.leftmost) return false
        if (rightmost != other.rightmost) return false
        if (citiesLeft != other.citiesLeft) return false

        return true
    }

    override fun hashCode(): Int {
        var result = leftmost.hashCode()
        result = 31 * result + rightmost.hashCode()
        result = 31 * result + citiesLeft.hashCode()
        return result
    }
}

fun movePrimeToTen(kdTree: KDTree) {
    for (city in kdTree.pcities) {
        if (!city.isPrime || city.pathindex % 10 == 9) {continue}
        val nothercity = kdTree.findNearestTen(city, 1).poll().second
        val citycurrindex = city.pathindex
        city.pathindex = nothercity.pathindex
        nothercity.pathindex = citycurrindex
        kdTree.pcities[city.pathindex] = city
        kdTree.pcities[nothercity.pathindex] = nothercity
        println(citycurrindex)
    }
}

class LeftoverComplicated(val leftmost: Set<City>, val rightmost: Set<City>,
                          val citiesLeft: Set<City>,
                          val closestCities: Map<City, List<Pair<Double, City>>>, val leftovernum:Int,
                          val firstSecondCity: MutableMap<City, Pair<Int,Int>>,
                          val prevcity:City, val tensleft: Int, val previndex: Int) {
    var heuristicminimum: Double = Double.POSITIVE_INFINITY
    var origlowerbound: Double = Double.POSITIVE_INFINITY
    var otherlowerbound: Double = Double.POSITIVE_INFINITY
    fun heurmin(): Double {
//        println("citiesleft = ${citiesLeft.size}")
        if ( citiesLeft.size == 0) {return 0.0}
        if (heuristicminimum != Double.POSITIVE_INFINITY) {
            return heuristicminimum
        }
        var heurval = 0.0
        val cityusingcity = HashMap<City, MutableList<City>>()
        citiesLeft.forEach {cityusingcity.put(it, ArrayList())}
        for (city in leftmost.plus(rightmost)) {
            cityusingcity.put(city, ArrayList())
        }
//        var minoutgoing = PriorityQueue<Double>(Comparator<Double> {d1, d2 -> d2.compareTo(d1)})
        for (city in citiesLeft) {
            var taken = 0
            val neighbours = closestCities[city]!!
            var (firstindex, secondindex) = firstSecondCity[city]!!
            var currindex = firstindex
//            var nearestwasleftmostorrightmost = false
            while (true) {
                val neighbour = neighbours[currindex]
                if (leftmost.contains(neighbour.second) ||
                    rightmost.contains(neighbour.second) ||
                    citiesLeft.contains(neighbour.second)) {
                    taken++
                    heurval += neighbour.first
                    cityusingcity[neighbour.second]!!.add(city)
                    if (taken == 1) {
                        firstindex = currindex
                        currindex = max(currindex, secondindex - 1)
                    } else if (taken == 2) {
                        secondindex = currindex
                        break
                    }
                }
                currindex++
            }
            firstSecondCity.put(city, Pair(firstindex, secondindex))
        }
        for (city in leftmost.plus(rightmost)) {
            var (firstindex, secondindex) = firstSecondCity[city]!!
            var currindex = firstindex
            val neighbours = closestCities[city]!!
            while (true) {
                val neighbour = neighbours[currindex]
                if (citiesLeft.contains(neighbour.second)) {
                    firstindex = currindex
                    heurval += neighbour.first
                    cityusingcity[neighbour.second]!!.add(city)
                    break
                }
                currindex++
            }
            firstSecondCity.put(city, Pair(firstindex, firstindex))
        }
        heuristicminimum = (heurval/2)
        val pushtothirdcosts = HashMap<City, MutableList<Double>>()
        for (entry in cityusingcity.entries) {
            val maxedgesto = when (leftmost.contains(entry.key) ||
                    rightmost.contains(entry.key)) {
                true    -> 1
                false   -> 2
            }
            if (entry.value.size > maxedgesto) {pushtothirdcosts.put(entry.key, ArrayList())}
        }
        for (city in citiesLeft) {
            var usethird = false
            var replacefirst = false
            var replacesecond = false
            val closecities = closestCities[city]!!
            val (firstcityind, secondcityind) = firstSecondCity[city]!!
            val maxcitiesfirst = when (leftmost.contains(closecities[firstcityind].second) ||
                    rightmost.contains(closecities[firstcityind].second)) {
                true    -> 1
                false   -> 2
            }
            val maxcitiessecond = when (leftmost.contains(closecities[secondcityind].second) ||
                    rightmost.contains(closecities[secondcityind].second)) {
                true    -> 1
                false   -> 2
            }
            replacefirst = maxcitiesfirst < cityusingcity[closecities[firstcityind].second]!!.size
            replacesecond = maxcitiessecond < cityusingcity[closecities[secondcityind].second]!!.size
            usethird = replacefirst || replacesecond
            if (usethird) {
                var currindex = secondcityind+1
                val neighbours = closecities
                while (true) {
                    val neighbour = neighbours[currindex]
                    if (leftmost.contains(neighbour.second) || rightmost.contains(neighbour.second) ||
                        citiesLeft.contains(neighbour.second)) {
                        if (replacefirst) {pushtothirdcosts[closecities[firstcityind].second]!!.add(
                            closecities[currindex].first - closecities[firstcityind].first
                        )}
                        if (replacesecond) {pushtothirdcosts[closecities[secondcityind].second]!!.add(
                            closecities[currindex].first - closecities[secondcityind].first
                        )}
                        break
                    }
                    currindex++
                }
            }
        }
        for (city in leftmost.plus(rightmost)) {
            var replacefirst = false
            val closecities = closestCities[city]!!
            val (firstcityind, secondcityind) = firstSecondCity[city]!!
            val maxcitiesfirst = when (leftmost.contains(closecities[firstcityind].second) ||
                    rightmost.contains(closecities[firstcityind].second)) {
                true    -> 1
                false   -> 2
            }
            replacefirst = maxcitiesfirst < cityusingcity[closecities[firstcityind].second]!!.size
            if (replacefirst) {
                var currindex = secondcityind + 1
                val neighbours = closecities
                while (true) {
                    val neighbour = neighbours[currindex]
                    if (citiesLeft.contains(neighbour.second)) {
                        pushtothirdcosts[closecities[firstcityind].second]!!.add(
                            closecities[currindex].first - closecities[firstcityind].first)
                        break
                    }
                    currindex++
                }
            }
        }
        origlowerbound = heuristicminimum
        for (entry in pushtothirdcosts.entries) {
            entry.value.sort()
//            println(entry.value)
            val maxedgesto = when (leftmost.contains(entry.key) ||
                    rightmost.contains(entry.key)) {
                true    -> 1
                false   -> 2
            }
//            println("sublist ${entry.value.subList(0, entry.value.size-maxedgesto)} $maxedgesto")
            if (entry.value.size > maxedgesto) {
                for (v in entry.value.subList(0, entry.value.size - maxedgesto)) {
                    heuristicminimum += v / 2
                }
            }
        }
        otherlowerbound = heuristicminimum
        heuristicminimum = max(heuristicminimum, otherheurmin()) + calcprimepenalty()
        return heuristicminimum
    }

    fun otherheurmin():Double {
        var oheurval = 0.0
        if ( citiesLeft.size == 0) {return 0.0}
        var heurval = 0.0
        val cityusingcity = HashMap<City, MutableList<City>>()
        citiesLeft.forEach {cityusingcity.put(it, ArrayList())}
        for (city in rightmost) {
            cityusingcity.put(city, ArrayList())
        }
        val closestcityindex = HashMap<City, Int>()
//        var minoutgoing = PriorityQueue<Double>(Comparator<Double> {d1, d2 -> d2.compareTo(d1)})
        for (city in citiesLeft) {
            val neighbours = closestCities[city]!!
            var (firstindex, secondindex) = firstSecondCity[city]!!
            var currindex = firstindex
            while (true) {
                val neighbour = neighbours[currindex]
                if (rightmost.contains(neighbour.second) ||
                    citiesLeft.contains(neighbour.second)) {
                    heurval += neighbour.first
                    cityusingcity[neighbour.second]!!.add(city)
                    closestcityindex[city] = currindex
                    break
                }
                currindex++
            }
        }
        for (city in leftmost) {
            var (firstindex, secondindex) = firstSecondCity[city]!!
            var currindex = firstindex
            val neighbours = closestCities[city]!!
            while (true) {
                val neighbour = neighbours[currindex]
                if (citiesLeft.contains(neighbour.second)) {
                    heurval += neighbour.first
                    cityusingcity[neighbour.second]!!.add(city)
                    closestcityindex[city] = currindex
                    break
                }
                currindex++
            }
        }
        heuristicminimum = heurval
        val pushtothirdcosts = HashMap<City, MutableList<Double>>()
        for (entry in cityusingcity.entries) {
            val maxedgesto = 1
            if (entry.value.size > maxedgesto) {pushtothirdcosts.put(entry.key, ArrayList())}
        }
        for (city in citiesLeft) {
            var replacefirst = false
            val closecities = closestCities[city]!!
            val firstcityind = closestcityindex[city]!!
            replacefirst = 1 < cityusingcity[closecities[firstcityind].second]!!.size
            if (replacefirst) {
                var currindex = firstcityind+1
                val neighbours = closecities
                while (true) {
                    val neighbour = neighbours[currindex]
                    if (leftmost.contains(neighbour.second) || rightmost.contains(neighbour.second) ||
                        citiesLeft.contains(neighbour.second)) {
                        if (replacefirst) {pushtothirdcosts[closecities[firstcityind].second]!!.add(
                            closecities[currindex].first - closecities[firstcityind].first
                        )}
                        break
                    }
                    currindex++
                }
            }
        }
        for (city in leftmost) {
            var replacefirst = false
            val closecities = closestCities[city]!!
            val firstcityind = closestcityindex[city]!!
            replacefirst = 1 < cityusingcity[closecities[firstcityind].second]!!.size
            if (replacefirst) {
                var currindex = firstcityind + 1
                val neighbours = closecities
                while (true) {
                    val neighbour = neighbours[currindex]
                    if (citiesLeft.contains(neighbour.second)) {
                        pushtothirdcosts[closecities[firstcityind].second]!!.add(
                            closecities[currindex].first - closecities[firstcityind].first)
                        break
                    }
                    currindex++
                }
            }
        }
        origlowerbound = heuristicminimum
        for (entry in pushtothirdcosts.entries) {
            entry.value.sort()
//            println(entry.value)
            val maxedgesto = 1
//            println("sublist ${entry.value.subList(0, entry.value.size-maxedgesto)} $maxedgesto")
            if (entry.value.size > maxedgesto) {
                for (v in entry.value.subList(0, entry.value.size - maxedgesto)) {
                    heuristicminimum += v
                }
            }
        }
        return heuristicminimum
    }

    fun calcprimepenalty():Double {
        var primepenalty = 0.0
        val primesleftover = citiesLeft.count {it.isPrime}
        val tensuncovered = tensleft - primesleftover
        if (tensuncovered > 0) {
            val edgevals = ArrayList<Double>()
            for (city in citiesLeft) {
                if (city.isPrime) {continue}
                val neighbours = closestCities[city]!!
                var (firstindex, secondindex) = firstSecondCity[city]!!
                var currindex = firstindex
                while (true) {
                    val neighbour = neighbours[currindex]
                    if (rightmost.contains(neighbour.second) ||
                        citiesLeft.contains(neighbour.second)
                    ) {
                        edgevals.add(neighbour.first*0.1)
                        break
                    }
                    currindex++
                }
            }
            edgevals.sort()
            primepenalty += edgevals.subList(0, tensuncovered).sum()
        }
        for (city in leftmost) {
            if (city.isPrime) {continue}
            val isten = when(prevcity == city) {
                false   -> city.pathindex % 10 == 9
                true    -> previndex % 10 == 9
            }
            if (!isten) {continue}
            var (firstindex, secondindex) = firstSecondCity[city]!!
            var currindex = firstindex
            val neighbours = closestCities[city]!!
            while (true) {
                val neighbour = neighbours[currindex]
                if (citiesLeft.contains(neighbour.second)) {
                    primepenalty += neighbour.first * 0.1
                    break
                }
                currindex++
            }
        }
        return primepenalty
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as LeftoverComplicated

        if (prevcity != other.prevcity) return false
        if (citiesLeft != other.citiesLeft) return false

        return true
    }

    override fun hashCode(): Int {
        var result = leftmost.hashCode()
        result = 31 * result + rightmost.hashCode()
        result = 31 * result + citiesLeft.hashCode()
        return result
    }
}

fun removeLargestEdges(kdTree: KDTree, edgegreaterthan: Double) {
    val pcities = kdTree.pcities
    val edgepq = PriorityQueue<Triple<City, City, Double>>(Comparator<Triple<City,City,Double>>{
            t1, t2 -> t2.third.compareTo(t1.third)
    })
    edgepq.add(Triple(pcities.last(), pcities.first(), pcities.first().calcDistance(pcities.last())))
    //var largestedge = Triple(pcities.last(), pcities.first(), pcities.first().calcDistance(pcities.last()))
    for (i in 0..pcities.size-2) {
        val dist = pcities[i].calcDistance(pcities[i+1])
        if (dist >= edgegreaterthan) {
            edgepq.offer(Triple(pcities[i], pcities[i + 1], dist))
        }
    }
    while (edgepq.isNotEmpty()) {
        val (origcity, destcity, distance) = edgepq.poll()
        println(kdTree.findNearestNeighbours(origcity, 1).poll().second.calcDistance(origcity))
        if (origcity.nextCity(pcities) != destcity) {continue}
        val pathto = HashMap<City, cityLink>()
        val pq = PriorityQueue<cityLink>(Comparator<cityLink> {
                p1, p2 -> p1.cost.compareTo(p2.cost)
        })
        val initcitylink = cityLink(null, origcity, 0.0, 0.0)
        pq.offer(initcitylink)
        while (pq.isNotEmpty()) {
            println(pq.size)
            val cl = pq.poll()
            if (pathto.containsKey(cl.city)) {continue}
            pathto.put(cl.city, cl)
            if (cl.city == destcity) {println("found city");break}
            val neighbours = kdTree.findNearestNeighbours(cl.city, 100000, radius = 10.0)
//            println("${cl.city}, ${neighbours.size}, ${pathto.size}, ${neighbours.peek().second.calcDistance(cl.city)}")
            for ((dist, ncity) in neighbours) {
                pq.offer(cityLink(cl, ncity, ncity.calcDistance(cl.city)+dist, 0.0))
            }
        }
        val cityLink = pathto[destcity]
        if (cityLink == null) {
            println("Can't find path to with edge ${distance}")
        } else {
            println("found path for edge ${distance}")
            val newrun = ArrayList<City>()
            var curlink = cityLink
            while (curlink != null) {
                newrun.add(curlink.city)
                curlink = curlink.parent
            }
            val removedcities = HashSet<City>(newrun)
            newrun.reverse()
            val neworder = ArrayList<City>()
            for (i in 0..pcities.size-1) {
                if (pcities[i] == newrun[0]) {
                    neworder.addAll(newrun)
                }
                if (!removedcities.contains(pcities[i])) {
                    neworder.add(pcities[i])
                }
            }
            for (i in 0..neworder.size-1) {
                kdTree.pcities[i] = neworder[i]
                neworder[i].pathindex = i
            }
        }
    }
    kdTree.updateMaxOutGoingRecursive()
}

fun DynamicPathSearch2(pcities:Array<City>, citiestoreplace: List<City>,
                       leftmost: HashSet<City>, rightmost: HashSet<City>,
                       maxfringe: Int = 100) {
    val only = 5
    val closestCities = HashMap<City, List<Pair<Double, City>>>()
    var origcost = 0.0
    for (i in 0..citiestoreplace.size-1) {
        val city = citiestoreplace[i]
        val pi = city.pathindex
        if (i == 0 || citiestoreplace[i-1].pathindex != city.pathindex-1) {
            var prevcost = city.distPrevCity(pcities)
            if (!pcities[pi-1].isPrime && pi % 10 == 0) {
                prevcost *= 1.1
            }
            origcost += prevcost
        }
        if (i == citiestoreplace.size-1 ||
            citiestoreplace[i+1].pathindex != city.pathindex+1) {
        }
        var nextcost = city.distNextCity(pcities)
        if (!pcities[pi].isPrime && pi % 10 == 9) {
            nextcost *= 1.1
        }
        origcost += nextcost
    }
    val cities = citiestoreplace
    val firstSecondCity = HashMap<City, Pair<Int,Int>>()
    for (city in cities) {
        val neighbours = ArrayList(cities.filter {
            it != city
        }.map { Pair(it.calcDistance(city), it)})
        neighbours.addAll(leftmost.map {Pair(it.calcDistance(city), it)})
        neighbours.addAll(rightmost.map {Pair(it.calcDistance(city), it)})
        neighbours.sortBy { it.first }
        closestCities.put(city, neighbours)
        firstSecondCity.put(city, Pair(0,0))
    }
    for (city in leftmost.plus(rightmost)) {
        val neighbours = ArrayList(cities.map { Pair(it.calcDistance(city), it)})
        neighbours.sortBy { it.first }
        closestCities.put(city, neighbours)
        firstSecondCity.put(city, Pair(0,0))
    }
    var tensleft = cities.count {it.pathindex % 10 == 9}
    val lo = LeftoverComplicated(leftmost, rightmost,
        HashSet(cities), closestCities, 0,
        HashMap(firstSecondCity), citiestoreplace.first().prevCity(pcities),
        tensleft, citiestoreplace.first().prevCity(pcities).pathindex)
//    println("current $origcost vs lowerbound ${lo.heurmin()} vs origlower bound ${lo.origlowerbound}" +
//            " ${lo.otherlowerbound}")
    var pq = TreeSet<Pair<LeftoverComplicated, cityLink>>(Comparator<Pair<LeftoverComplicated, cityLink>> {
            p1, p2 ->
        when (p2.second.cost.compareTo(p1.second.cost)) {
            0   -> p1.first.leftovernum.compareTo(p2.first.leftovernum)
            else-> p1.second.cost.compareTo(p2.second.cost)
        }
    })
    pq.add(Pair(lo, cityLink(null, pcities[0], 0.0, lo.heurmin())))
//    println(cities.map {it.pathindex})
    for (i in 0..cities.size-1) {
        //println("$i ${pq.first().first.citiesLeft.size}")
        val newleftovermap = HashMap<LeftoverComplicated, Pair<LeftoverComplicated, cityLink>>()
        val newpq = TreeSet<Pair<LeftoverComplicated, cityLink>>(
            Comparator<Pair<LeftoverComplicated, cityLink>> {
                    p1, p2 ->
                val firstcompare =
                    (p1.first.heurmin() + p1.second.cost).compareTo(
                        p2.first.heurmin() + p2.second.cost
                    )
                when (firstcompare) {
                    0   -> p1.first.leftovernum.compareTo(p2.first.leftovernum)
                    else-> firstcompare
                }
            })
        val pi = cities[i].pathindex
        val rrem = when (i == cities.size-1 || cities[i+1].pathindex -1 != pi) {
            true    -> cities[i].nextCity(pcities)
            false   -> null
        }
        if (pi % 10 == 9) {tensleft--}
//        println("step $i lowerbound = ${pq.first().second.lowerbound} ${pq.size}")
//        println(pq.first().first.citiesLeft.size)
        for (prevassign in pq) {
//            println(prevassign.first.leftmost)
//            println(prevassign.first.prevcity)
            val citiestoconsider = ArrayList<City>()
            for (city in closestCities[prevassign.first.prevcity]!!) {
                if (prevassign.first.citiesLeft.contains(city.second)) {
                    citiestoconsider.add(city.second)
                    if (citiestoconsider.size >= only) {
                        break
                    }
                }
            }

//            citiestoconsider.addAll(prevassign.first.citiesLeft)
            for (city in citiestoconsider) {
                val newleftmost = HashSet(prevassign.first.leftmost)
                val newrightmost = HashSet(prevassign.first.rightmost)
                newleftmost.remove(prevassign.first.prevcity)
                if (rrem != null) {
                    newrightmost.remove(rrem)
                }
                var edgecost:Double = prevassign.first.prevcity.calcDistance(city)
                if ((cities[i].pathindex-1)%10 == 9 && !prevassign.first.prevcity.isPrime) {
                    edgecost *= 1.1
                }
                var currcost = prevassign.second.cost + edgecost
//                println("first add: $currcost")
                val prevcity:City
                val previndex:Int
//                println(rrem)
                if (rrem != null) {
                    var edgecost:Double = rrem.calcDistance(city)
                    if ((cities[i].pathindex)%10 == 9 && !city.isPrime) {
                        edgecost *= 1.1
                    }
                    currcost += edgecost
                    if (i == cities.size-1) {
                        prevcity = pcities[0]
                        previndex = 0
                    } else {
                        prevcity = cities[i+1].prevCity(pcities)
                        previndex = prevcity.pathindex
                    }
//                    println("second add: $currcost")
                } else {
                    newleftmost.add(city)
                    prevcity = city
                    previndex = citiestoreplace[i].pathindex
                }
                val newleftover = LeftoverComplicated(newleftmost, newrightmost,
                    prevassign.first.citiesLeft.minus(city),
                    closestCities, newleftovermap.size,
                    prevassign.first.firstSecondCity.toMutableMap(),
                    prevcity, tensleft, previndex)
                val currassign = newleftovermap.get(newleftover)
//                println(currcost)
                if (currassign != null) {
                    if (currassign.second.cost <= currcost) {continue}
                    newpq.remove(currassign)
                    val newassign = Pair(currassign.first, cityLink(prevassign.second, city,
                        currcost, currcost + currassign.first.heurmin()))
                    newpq.add(newassign)
                    newleftovermap.put(newassign.first, newassign)
                } else {
                    if (newleftover.heurmin() + currcost >= origcost) {continue}
                    val newassign = Pair(newleftover, cityLink(prevassign.second, city, currcost,
                        newleftover.heurmin() + currcost))
                    if (newpq.size >= maxfringe && newpq.last().second.lowerbound < newassign.second.lowerbound) {continue}
                    newpq.add(newassign)
                    newleftovermap.put(newassign.first, newassign)
                    while (newpq.size > maxfringe) {
                        val toremove = newpq.pollLast()
                        newleftovermap.remove(toremove.first)
                    }
                }
            }
        }
        pq = newpq
    }
    if (pq.isEmpty()) {
        println("No result")
    } else {
        val bestone = pq.first()
        var newcost = bestone.second.cost
        println("${newcost} vs $origcost")
        if (newcost >= origcost) {return}
        //println("Cities Left ${bestone.first.citiesLeft.size}")
        val cityindices = citiestoreplace.map { it.pathindex }
        val initialpathcost = calcPathWeight(pcities, true)
        var i = citiestoreplace.size-1
        var currlink = bestone.second
        while (true) {
            val pi = cityindices[i]
            pcities[pi] = currlink.city
            currlink.city.pathindex = pi
            if (i == 0) {
                break
            } else {
                i--
                currlink = currlink.parent!!
            }
        }
        val newpathcost = calcPathWeight(pcities, true)
        println("$initialpathcost vs $newpathcost")
    }
}

fun calcminspanningtree(cities:Set<City>,
                        closestCities: Map<City, List<Pair<Double, City>>>,
                        firstSecondCity: MutableMap<City, Pair<Int, Int>>):Double {
    val pq = PriorityQueue<Triple<Double, City, City>>(Comparator<Triple<Double, City, City>> {
            p1, p2 -> p1.first.compareTo(p2.first)
    })
    var mspdist = 0.0
    val assignedcities = HashSet<City>()
    assignedcities.add(cities.first())
    val firstindex = HashMap<City, Int>()
    for (city in cities) {
        firstindex[city] = firstSecondCity[city]!!.first
    }
    fun addClosestEdge(city:City) {
        var ind = firstindex[city]!!
        val neighbours = closestCities[city]!!
        while (assignedcities.contains(neighbours[ind].second) ||
            !cities.contains(neighbours[ind].second)) {
            ind++
        }
        firstindex[city] = ind
        val (dist, succ) = neighbours[ind]
        pq.add(Triple(dist, city, succ))
    }
    addClosestEdge(assignedcities.first())
    while (assignedcities.size < cities.size) {
        val (dist, city1, city2) = pq.poll()
        if (assignedcities.add(city2)) {
            mspdist += dist
            if (assignedcities.size == cities.size) {break}
            addClosestEdge(city2)
        }
        addClosestEdge(city1)
    }
    return mspdist
}