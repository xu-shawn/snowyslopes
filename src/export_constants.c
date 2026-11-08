/* Build-time data extraction only; the standalone finder does not link Cubiomes. */
#include "biomenoise.h"
#include "tables/btree263.h"
#include <stdio.h>

static const Spline *nodes[512];
static int count;
static int visit(const Spline *p) {
    for(int i=0;i<count;i++) if(nodes[i]==p) return i;
    int id=count++; nodes[id]=p;
    if(p->len>1) for(int i=0;i<p->len;i++) visit(p->val[i]);
    return id;
}
int main(void) {
    BiomeNoise bn; initBiomeNoise(&bn,MC_26_3); visit(bn.sp);
    puts("// Generated from xpple/cubiomes 18edd56575a60fe7129705bf972de0a511437527.\n// Minecraft 26.3 biome parameter and terrain spline DATA. See THIRD_PARTY.md.\n#pragma once\n#include <cstdint>\nnamespace standalone {\nstruct SplineData { int axis,n; float value; float x[12],slope[12]; int child[12]; };\ninline constexpr SplineData splines[] = {");
    for(int i=0;i<count;i++) {
        const Spline *p=nodes[i];
        if(p->len==1) { printf("{-1,0,%af,{},{},{}},\n",((FixSpline*)p)->val); continue; }
        printf("{%d,%d,0,{",p->typ,p->len);
        for(int k=0;k<p->len;k++) printf("%s%af",k?",":"",p->loc[k]);
        printf("},{");
        for(int k=0;k<p->len;k++) printf("%s%af",k?",":"",p->der[k]);
        printf("},{");
        for(int k=0;k<p->len;k++) printf("%s%d",k?",":"",visit(p->val[k]));
        puts("}},");
    }
    puts("};\ninline constexpr int32_t parameters[][2] = {");
    for(unsigned i=0;i<sizeof(btree263_param)/sizeof(*btree263_param);i++)
        printf("{%d,%d},\n",btree263_param[i][0],btree263_param[i][1]);
    puts("};\ninline constexpr uint64_t tree[] = {");
    for(unsigned i=0;i<sizeof(btree263_nodes)/sizeof(*btree263_nodes);i++)
        printf("0x%016" PRIx64 "ULL,\n",btree263_nodes[i]);
    puts("};\ninline constexpr int steps[] = {1555,259,43,7,1,0};\n}");
}
