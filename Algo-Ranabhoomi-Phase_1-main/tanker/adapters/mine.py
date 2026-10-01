
"""High-quality CVRP heuristic for Tanker Run.

Uses several complementary constructions and capacity-aware route local search:
Clarke-Wright savings, angle sweep, cheapest insertion, 2-opt, relocate,
swap, and cross-route 2-opt*. It keeps the best valid plan and reports
improvements throughout the budget.
"""
import time, math, random

from adapter import Solver
from data import distance_matrix


class MySolver(Solver):
    def solve(self, instance, submit_candidate):
        start = time.perf_counter()
        deadline = start + 5.82
        d = distance_matrix(instance)
        n = instance.size
        Q = instance.capacity
        dem = instance.demand
        K = instance.fleet
        best_cost = 10**18
        best_routes = None
        last_submit = 0.0

        def cost_route(r):
            if not r: return 0
            s = d[0][r[0]] + d[r[-1]][0]
            for a,b in zip(r,r[1:]): s += d[a][b]
            return s

        def total(rs):
            return sum(cost_route(r) for r in rs)

        def loads(rs):
            return [sum(dem[v] for v in r) for r in rs]

        def clean(rs):
            return [r for r in rs if r]

        def publish(rs, force=False):
            nonlocal best_cost, best_routes, last_submit
            now = time.perf_counter()
            if now >= deadline and not force:
                return
            rs = clean(rs)
            c = total(rs)
            if c < best_cost:
                best_cost = c
                best_routes = [r[:] for r in rs]
                submit_candidate({"routes": best_routes})
                last_submit = now

        # ---- constructions ----
        def savings_construct():
            rs = [[v] for v in range(1,n+1)]
            ld = [dem[v] for v in range(1,n+1)]
            where = {v:i for i,v in enumerate(range(1,n+1))}
            savings=[]
            for i in range(1,n+1):
                di=d[0][i]
                for j in range(i+1,n+1):
                    savings.append((di+d[0][j]-d[i][j],i,j))
            savings.sort(reverse=True)
            alive=[True]*len(rs)
            for s,i,j in savings:
                if time.perf_counter() >= deadline: break
                ai=where[i]; aj=where[j]
                if ai==aj or not alive[ai] or not alive[aj]:
                    continue
                if ld[ai]+ld[aj] > Q:
                    continue
                A=rs[ai]; B=rs[aj]
                # orient so i is an endpoint and j is an endpoint
                if A[-1] != i:
                    if A[0] == i: A.reverse()
                    else: continue
                if B[0] != j:
                    if B[-1] == j: B.reverse()
                    else: continue
                A.extend(B)
                ld[ai]+=ld[aj]
                alive[aj]=False
                for v in B: where[v]=ai
                # B contents are now also in A; no further use of B object
            return [rs[i] for i in range(len(rs)) if alive[i]]

        def sweep_construct(offset=0.0):
            ang=[]
            cx,cy=instance.coords[0]
            for v in range(1,n+1):
                x,y=instance.coords[v]
                a=math.atan2(y-cy,x-cx)-offset
                ang.append((a%(2*math.pi),v))
            ang.sort()
            rs=[]; r=[]; ld=0
            for _,v in ang:
                if r and ld+dem[v]>Q:
                    rs.append(r); r=[]; ld=0
                r.append(v); ld+=dem[v]
            if r: rs.append(r)
            return rs

        def insertion_construct():
            # Start with geographically far villages; insert each remaining
            # village at its cheapest feasible position.
            remaining=set(range(1,n+1))
            rs=[]
            # Seed enough routes to avoid a bad single cluster swallowing capacity.
            seeds=sorted(remaining,key=lambda v:d[0][v],reverse=True)
            target=max(1,min(K, max(2,n//8)))
            for v in seeds[:target]:
                rs.append([v]); remaining.remove(v)
            ld=[dem[r[0]] for r in rs]
            while remaining:
                bv=None; br=bi=None; gain=10**18
                # cheapest insertion, sampled by all candidates for n<=150
                for v in remaining:
                    for a,r in enumerate(rs):
                        if ld[a]+dem[v]>Q: continue
                        for j in range(len(r)+1):
                            x=r[j-1] if j else 0
                            y=r[j] if j<len(r) else 0
                            g=d[x][v]+d[v][y]-d[x][y]
                            if g<gain:
                                gain=g; bv=v;br=a;bi=j
                    if time.perf_counter() >= deadline: return rs
                if bv is None:
                    v=min(remaining,key=lambda x:dem[x])
                    rs.append([v]); ld.append(dem[v]); remaining.remove(v)
                else:
                    rs[br].insert(bi,bv);ld[br]+=dem[bv];remaining.remove(bv)
            return rs

        def two_opt(rs):
            changed=True
            while changed and time.perf_counter()<deadline:
                changed=False
                for r in rs:
                    m=len(r)
                    if m<4: continue
                    for i in range(m-1):
                        a=r[i-1] if i else 0
                        for j in range(i+1,m):
                            b=r[j+1] if j+1<m else 0
                            old=d[a][r[i]]+d[r[j]][b]
                            new=d[a][r[j]]+d[r[i]][b]
                            if new<old:
                                r[i:j+1]=reversed(r[i:j+1])
                                changed=True
                                break
                        if changed: break
                    if changed: break
            return rs

        def relocate_best(rs, ld):
            best=None
            nr=len(rs)
            for a in range(nr):
                ra=rs[a]
                for i,v in enumerate(ra):
                    if time.perf_counter()>=deadline: return False
                    nv=dem[v]
                    p=ra[i-1] if i else 0
                    q=ra[i+1] if i+1<len(ra) else 0
                    remove=d[p][v]+d[v][q]-d[p][q]
                    for b in range(nr):
                        if a==b or ld[b]+nv>Q: continue
                        rb=rs[b]
                        for j in range(len(rb)+1):
                            x=rb[j-1] if j else 0
                            y=rb[j] if j<len(rb) else 0
                            add=d[x][v]+d[v][y]-d[x][y]
                            gain=remove-add
                            if gain>0 and (best is None or gain>best[0]):
                                best=(gain,a,i,b,j,v)
            if best:
                _,a,i,b,j,v=best
                rs[a].pop(i);rs[b].insert(j,v)
                ld[a]-=dem[v];ld[b]+=dem[v]
                return True
            return False

        def swap_best(rs,ld):
            best=None; nr=len(rs)
            for a in range(nr):
                ra=rs[a]
                for i,xv in enumerate(ra):
                    if time.perf_counter()>=deadline:return False
                    dx=dem[xv]
                    ax=ra[i-1] if i else 0
                    bx=ra[i+1] if i+1<len(ra) else 0
                    oldx=d[ax][xv]+d[xv][bx]
                    for b in range(a+1,nr):
                        rb=rs[b]
                        for j,yv in enumerate(rb):
                            if ld[a]-dx+dem[yv]>Q or ld[b]-dem[yv]+dx>Q: continue
                            ay=rb[j-1] if j else 0
                            by=rb[j+1] if j+1<len(rb) else 0
                            old=oldx+d[ay][yv]+d[yv][by]
                            new=d[ax][yv]+d[yv][bx]+d[ay][xv]+d[xv][by]
                            gain=old-new
                            if gain>0 and (best is None or gain>best[0]):
                                best=(gain,a,i,b,j)
            if best:
                _,a,i,b,j=best
                rs[a][i],rs[b][j]=rs[b][j],rs[a][i]
                ld[a]=ld[a]-dem[rs[b][j]]+dem[rs[a][i]] # corrected below
                # recompute two touched loads safely
                ld[a]=sum(dem[v] for v in rs[a]);ld[b]=sum(dem[v] for v in rs[b])
                return True
            return False


        def cross_two_opt(rs,ld):
            # Exchange tails after i/j. Prefix loads make this O(n^2) per route pair.
            nr=len(rs); best=None
            for a in range(nr):
                A=rs[a]
                pa=[0]
                for v in A: pa.append(pa[-1]+dem[v])
                for b in range(a+1,nr):
                    B=rs[b]
                    pb=[0]
                    for v in B: pb.append(pb[-1]+dem[v])
                    for i in range(len(A)):
                        if time.perf_counter()>=deadline:return False
                        ap=A[i-1] if i else 0
                        x=A[i]
                        for j in range(len(B)):
                            bp=B[j-1] if j else 0
                            y=B[j]
                            la=pa[i]+(pb[-1]-pb[j])
                            lb=pb[j]+(pa[-1]-pa[i])
                            if la>Q or lb>Q: continue
                            old=d[ap][x]+d[bp][y]
                            newc=d[ap][y]+d[bp][x]
                            gain=old-newc
                            if gain>0 and (best is None or gain>best[0]):
                                best=(gain,a,b,i,j)
            if best:
                _,a,b,i,j=best
                A=rs[a];B=rs[b]
                rs[a]=A[:i]+B[j:]
                rs[b]=B[:j]+A[i:]
                ld[a]=sum(dem[v] for v in rs[a]);ld[b]=sum(dem[v] for v in rs[b])
                return True
            return False

        def local_search(rs):
            # Remove empty routes, then repeatedly improve with strongest neighborhoods.
            rs=clean(rs)
            ld=loads(rs)
            two_opt(rs)
            while time.perf_counter()<deadline-0.12:
                improved=False
                if relocate_best(rs,ld):
                    two_opt(rs); improved=True
                elif swap_best(rs,ld):
                    two_opt(rs); improved=True
                elif cross_two_opt(rs,ld):
                    two_opt(rs); improved=True
                if not improved: break
            return rs


        def route_first_split():
            # Build a depot-rooted giant tour, then optimally cut it into
            # capacity-feasible routes by dynamic programming.
            variants=[]
            for mode in range(4):
                if time.perf_counter()>=deadline: break
                rem=set(range(1,n+1)); tour=[]
                if mode==0:
                    cur=0
                    while rem:
                        v=min(rem,key=lambda x:(d[cur][x],x))
                        tour.append(v);rem.remove(v);cur=v
                elif mode==1:
                    # farthest-first insertion: useful when clusters are tight
                    first=max(rem,key=lambda x:(d[0][x],x)); tour=[first];rem.remove(first)
                    while rem:
                        v=max(rem,key=lambda x:min(d[x][0],*(d[x][u] for u in tour)))
                        bestj=0;bestg=10**18
                        for j in range(len(tour)+1):
                            a=tour[j-1] if j else 0;b=tour[j] if j<len(tour) else 0
                            g=d[a][v]+d[v][b]-d[a][b]
                            if g<bestg:bestg=g;bestj=j
                        tour.insert(bestj,v);rem.remove(v)
                else:
                    # angular tours with different rotations
                    ang=[]
                    cx,cy=instance.coords[0]
                    off=(mode-2)*math.pi/5
                    for v in range(1,n+1):
                        x,y=instance.coords[v]
                        ang.append(((math.atan2(y-cy,x-cx)-off)%(2*math.pi),v))
                    ang.sort();tour=[v for _,v in ang]
                # TSP 2-opt, but cap the work so later split/local search gets time.
                changed=True
                passes=0
                while changed and passes<3 and time.perf_counter()<deadline:
                    changed=False;passes+=1
                    for i in range(n-1):
                        a=tour[i-1] if i else 0
                        for j in range(i+1,n):
                            b=tour[j+1] if j+1<n else 0
                            if d[a][tour[j]]+d[tour[i]][b] < d[a][tour[i]]+d[tour[j]][b]:
                                tour[i:j+1]=reversed(tour[i:j+1]);changed=True;break
                        if changed:break
                # Exact shortest split for this tour, with route count naturally <= K.
                inf=10**18
                dp=[inf]*(n+1);prev=[-1]*(n+1);dp[0]=0
                for i in range(1,n+1):
                    load=0
                    for j in range(i,0,-1):
                        load+=dem[tour[j-1]]
                        if load>Q: break
                        seg=tour[j-1:i]
                        c=d[0][seg[0]]+d[seg[-1]][0]
                        for u,v in zip(seg,seg[1:]):c+=d[u][v]
                        val=dp[j-1]+c
                        if val<dp[i]:
                            dp[i]=val;prev[i]=j-1
                    if time.perf_counter()>=deadline: break
                if dp[n]<inf:
                    rs=[];i=n
                    while i:
                        j=prev[i];rs.append(tour[j:i]);i=j
                    variants.append(rs[::-1])
            return variants
        # Initial candidates.
        candidates=[]
        candidates.append(savings_construct())
        if time.perf_counter()<deadline-1.0:
            candidates.extend(route_first_split())
        if time.perf_counter()<deadline-0.8:
            candidates.append(sweep_construct(0))
        if time.perf_counter()<deadline-0.8:
            candidates.append(sweep_construct(math.pi/6))
        if time.perf_counter()<deadline-1.0:
            candidates.append(insertion_construct())

        for rs in candidates:
            if time.perf_counter()>=deadline: break
            rs=local_search([r[:] for r in rs])
            publish(rs)

        # Iterated destroy/repair. Regret-2 insertion is particularly useful
        # when capacity is tight because it protects villages with few options.
        rng=random.Random((instance.size<<20) ^ instance.capacity ^ sum(dem))
        while time.perf_counter()<deadline-0.12 and best_routes:
            rs=[r[:] for r in best_routes]
            allv=[v for r in rs for v in r]
            if len(allv)<8: break
            # Vary destruction size; larger removals help when routes are badly paired.
            frac=rng.choice((0.08,0.12,0.16,0.20))
            remove_n=min(24,max(5,int(len(allv)*frac)))
            chosen=rng.sample(allv,remove_n)
            chosen_set=set(chosen)
            newrs=[]
            for r in rs:
                nr=[v for v in r if v not in chosen_set]
                if nr:newrs.append(nr)
            rs=newrs
            ld=loads(rs)

            remaining=set(chosen)
            while remaining and time.perf_counter()<deadline-0.08:
                selected=None
                # regret-2: choose the village whose second-best position is
                # much worse than its best position.
                for v in remaining:
                    best1=None;best2=None
                    for a,r in enumerate(rs):
                        if ld[a]+dem[v]>Q: continue
                        for j in range(len(r)+1):
                            x=r[j-1] if j else 0
                            y=r[j] if j<len(r) else 0
                            g=d[x][v]+d[v][y]-d[x][y]
                            item=(g,a,j)
                            if best1 is None or g<best1[0]:
                                best2=best1;best1=item
                            elif best2 is None or g<best2[0]:
                                best2=item
                    if best1 is None:
                        regret=10**9
                    else:
                        regret=(best2[0]-best1[0]) if best2 else 10**6
                    # Small deterministic/random tie break favors large demand.
                    key=(regret,dem[v],-v)
                    if selected is None or key>selected[0]:
                        selected=(key,v,best1)
                _,v,ins=selected
                if ins is None:
                    if len(rs)>=K: break
                    rs.append([v]);ld.append(dem[v])
                else:
                    _,a,j=ins
                    rs[a].insert(j,v);ld[a]+=dem[v]
                remaining.remove(v)
            rs=local_search(rs)
            publish(rs)

        if best_routes is None:
            # Should never happen, but return a feasible fallback.
            rs=savings_construct()
            best_routes=clean(rs)
        return {"routes": best_routes}
