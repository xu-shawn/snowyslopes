#pragma once
// Independent Minecraft 26.3 normal Overworld sampler. No Cubiomes calls/types.
// Fixed arrays and integer offsets intentionally keep the kernel portable to CUDA.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include "worldgen_data.hpp"

namespace standalone {
inline uint64_t mix(uint64_t x) {
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
struct Random {
    uint64_t a,b;
    explicit Random(uint64_t seed) {
        seed ^= 0x6a09e667f3bcc909ULL;
        a=mix(seed); b=mix(seed+0x9e3779b97f4a7c15ULL);
    }
    Random(uint64_t a_,uint64_t b_):a(a_),b(b_){}
    uint64_t next() {
        uint64_t result=std::rotl(a+b,17)+a;
        b^=a; a=std::rotl(a,49)^b^(b<<21); b=std::rotl(b,28);
        return result;
    }
    double uniform() { return (next()>>11)*0x1p-53; }
    uint32_t bounded(uint32_t n) {
        uint64_t r=uint64_t(uint32_t(next()))*n;
        if(uint32_t(r)<n) while(uint32_t(r)<uint32_t(-n)%n)
            r=uint64_t(uint32_t(next()))*n;
        return r>>32;
    }
};
inline double fade(double t) { return t*t*t*(t*(t*6.-15.)+10.); }
template<class T> inline T interpolate(T t,T a,T b) { return a+t*(b-a); }
inline double gradient(unsigned h,double x,double y,double z) {
    // Minecraft's 16 gradient vectors, including its four duplicates. Components are 0/+-1, so
    // the products are exact and adding the zero term never changes a nonzero two-term sum:
    // branch-free but bit-identical to the two-term forms.
    static constexpr double gx[16]={1,-1,1,-1,1,-1,1,-1,0,0,0,0,1,0,-1,0};
    static constexpr double gy[16]={1,1,-1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1};
    static constexpr double gz[16]={0,0,0,0,1,1,-1,-1,1,1,-1,-1,0,1,0,-1};
    h&=15;
    return gx[h]*x+gy[h]*y+gz[h]*z;
}
struct Perlin {
    std::array<uint8_t,256> permutation;
    double ox,oy,oz,frequency,weight;
    double fy,vy;
    int iy;
    void init(Random r,double f,double w) {
        ox=r.uniform()*256; oy=r.uniform()*256; oz=r.uniform()*256;
        frequency=f; weight=w;
        for(int i=0;i<256;i++) permutation[i]=i;
        for(int i=0;i<256;i++) std::swap(permutation[i],permutation[i+r.bounded(256-i)]);
        iy=int(std::floor(oy)); fy=oy-std::floor(oy); vy=fade(fy);
    }
    double sample(double x,double y,double z) const {
        x+=ox; z+=oz;
        double bx=std::floor(x),bz=std::floor(z);
        int ix=int(bx),iz=int(bz),jy=iy;
        x-=bx; z-=bz;
        double dy=fy,ty=vy;
        if(y!=0) { y+=oy; double by=std::floor(y); jy=int(by);dy=y-by;ty=fade(dy); }
        auto hash=[&](int dx,int dyy,int dz) {
            int a=permutation[(ix+dx)&255];
            int b=permutation[(a+jy+dyy)&255];
            return permutation[(b+iz+dz)&255];
        };
        double tx=fade(x),tz=fade(z);
        double zlo=interpolate(ty,
            interpolate(tx,gradient(hash(0,0,0),x,dy,z),gradient(hash(1,0,0),x-1,dy,z)),
            interpolate(tx,gradient(hash(0,1,0),x,dy-1,z),gradient(hash(1,1,0),x-1,dy-1,z)));
        double zhi=interpolate(ty,
            interpolate(tx,gradient(hash(0,0,1),x,dy,z-1),gradient(hash(1,0,1),x-1,dy,z-1)),
            interpolate(tx,gradient(hash(0,1,1),x,dy-1,z-1),gradient(hash(1,1,1),x-1,dy-1,z-1)));
        return interpolate(tz,zlo,zhi);
    }
};
// MD5-derived positional RNG salts for octave_-10 through octave_0.
inline constexpr uint64_t octave_salt[][2]={
 {0x36d326eed40efeb2ULL,0x5be9ce18223c636aULL},{0x082fe255f8be6631ULL,0x4e96119e22dedc81ULL},
 {0x0ef68ec68504005eULL,0x48b6bf93a2789640ULL},{0xf11268128982754fULL,0x257a1d670430b0aaULL},
 {0xe51c98ce7d1de664ULL,0x5f9478a733040c45ULL},{0x6d7b49e7e429850aULL,0x2e3063c622a24777ULL},
 {0xbd90d5377ba1b762ULL,0xc07317d419a7548dULL},{0x53d39c6752dac858ULL,0xbcd1c5a80ab65b3eULL},
 {0xb4a24d7a84e7677bULL,0x023ff9668e89b5c4ULL},{0xdffa22b534c5f608ULL,0xb9b67517d3665ca9ULL},
 {0xd50708086cef4d7cULL,0x6e1651ecc7f43309ULL}};
enum Channel { Temperature,Humidity,Continentalness,Erosion,Shift,Weirdness };
struct Config { uint64_t a,b; int first,len; double amplitude[9]; };
inline constexpr Config configs[]={
 {0x5c7e6b29735f0d7fULL,0xf7d86f1bbc734988ULL,-10,6,{1.5,0,1,0,0,0}},
 {0x81bb4d22e8dc168eULL,0xf1c8b4bea16303cdULL,-8,6,{1,1,0,0,0,0}},
 {0x83886c9d0ae3a662ULL,0xafa638a61b42e8adULL,-9,9,{1,1,2,2,2,1,1,1,1}},
 {0xd02491e6058f6fd8ULL,0x4792512c94c17a80ULL,-9,5,{1,1,0,1,1}},
 {0x080518cf6af25384ULL,0x3f3dfb40a54febd5ULL,-3,4,{1,1,1,0}},
 {0xefc8ef4d36102b34ULL,0x1beeeb324a0f24eaULL,-7,6,{1,2,1,0,0,0}}};
// A heuristic only: favorable vertical phases allow broad continentalness
// extrema. Adapted as an experiment from COMMISSION's seed screening idea.
// This costs a handful of RNG operations, with no permutation construction.
inline double phase_score(uint64_t seed,double stop=1.) {
    Random world(seed);uint64_t a=world.next(),b=world.next();
    Random c(a^configs[Continentalness].a,b^configs[Continentalness].b);
    uint64_t fa[2],fb[2];
    for(int side=0;side<2;side++) {fa[side]=c.next();fb[side]=c.next();}
    double score=0;constexpr double weights[]={.35,.11,.04};
    for(int octave=0;octave<3;octave++) for(int side=0;side<2;side++) {
        const auto &salt=octave_salt[octave+1];
        Random r(fa[side]^salt[0],fb[side]^salt[1]);r.next();
        double y=r.uniform()*256.;
        score+=weights[octave]*std::abs((y-std::floor(y))-.5);
        if(score>=stop) return score;
    }
    return score;
}
struct NormalNoise {
    Perlin octave[2][9];
    int n=0;
    double factor=0;
    void init(uint64_t a,uint64_t b,Channel channel) {
        const auto &c=configs[channel]; Random r(a^c.a,b^c.b);
        for(int side=0;side<2;side++) {
            uint64_t ra=r.next(),rb=r.next(); n=0;
            double f=std::ldexp(1.,c.first);
            double w=double(1<<(c.len-1))/((1<<c.len)-1);
            for(int i=0;i<c.len;i++,f*=2,w*=.5) if(c.amplitude[i]) {
                auto &salt=octave_salt[c.first+i+10];
                octave[side][n++].init(Random(ra^salt[0],rb^salt[1]),f,w*c.amplitude[i]);
            }
        }
        int lo=0,hi=c.len-1;
        while(!c.amplitude[lo]) lo++;
        while(!c.amplitude[hi]) hi--;
        int span=hi-lo+1;
        factor=(5.*span)/(3.*(span+1));
    }
    double sample(double x,double y,double z) const {
        double sums[2]={};
        for(int side=0;side<2;side++) {
            double scale=side?337./331.:1.;
            double px=x*scale,py=y*scale,pz=z*scale;
            for(int j=0;j<n;j++) {
                const auto &o=octave[side][j];
                sums[side]+=o.weight*o.sample(px*o.frequency,py*o.frequency,pz*o.frequency);
            }
        }
        return (sums[0]+sums[1])*factor;
    }
};

inline float spline(int index,const float *v) {
    const auto &s=splines[index];
    if(s.n==0) return s.value;
    float x=v[s.axis]; int hi=0;
    while(hi<s.n && x>s.x[hi]) hi++;
    if(hi==0 || hi==s.n) {
        int j=hi==0?0:s.n-1;
        return spline(s.child[j],v)+s.slope[j]*(x-s.x[j]);
    }
    int lo=hi-1;
    float width=s.x[hi]-s.x[lo],t=(x-s.x[lo])/width;
    float a=spline(s.child[lo],v),b=spline(s.child[hi],v);
    float p=s.slope[lo]*width-(b-a),q=-s.slope[hi]*width+(b-a);
    return interpolate(double(t),double(a),double(b))
        + (t*(1.f-t))*interpolate(double(t),double(p),double(q));
}
inline uint64_t distance(int node,const int64_t *v) {
    uint64_t encoded=tree[node],sum=0;
    for(int i=0;i<6;i++) {
        const auto &r=parameters[(encoded>>(8*i))&255];
        int64_t d=std::max({int64_t(0),int64_t(r[0])-v[i],v[i]-int64_t(r[1])});
        sum+=uint64_t(d*d);
    }
    return sum;
}
inline int nearest(int node,int level,const int64_t *v,int best,uint64_t &best_distance) {
    if((tree[node]>>56)==255) {
        uint64_t d=distance(node,v);
        if(d<best_distance) {best_distance=d;best=node;}
        return best;
    }
    int child=tree[node]>>48;
    // Compressed levels use the next smaller stride where this node has fewer descendants.
    int step;
    do { step=steps[level++]; } while(node+step>=int(std::size(tree)));
    for(int i=0;i<6 && child<int(std::size(tree));i++,child+=step)
        if(distance(child,v)<best_distance) best=nearest(child,level,v,best,best_distance);
    return best;
}
inline int classify(const int64_t *v) {
    uint64_t d=std::numeric_limits<uint64_t>::max();
    return (tree[nearest(0,0,v,0,d)]>>48)&255;
}
inline bool closer_competitor(int node,int level,const int64_t *v,uint64_t limit,bool &tie) {
    uint64_t d=distance(node,v);
    if(node!=0 && d>limit) return false;
    if((tree[node]>>56)==255) {
        if(((tree[node]>>48)&255)==174) return false;
        if(d<limit) return true;
        tie=true;return false;
    }
    int stride;
    do {stride=steps[level++];} while(node+stride>=int(std::size(tree)));
    int child=tree[node]>>48;
    for(int i=0;i<6 && child<int(std::size(tree));i++,child+=stride)
        if(closer_competitor(child,level,v,limit,tie)) return true;
    return false;
}
inline int64_t interval_distance(int64_t value,int64_t lo,int64_t hi) {
    return std::max({int64_t(0),lo-value,value-hi});
}
struct Sampler {
    uint64_t a,b;
    NormalNoise noise[6];
    bool ready[6]={};
    explicit Sampler(uint64_t seed) { Random r(seed);a=r.next();b=r.next(); }
    const NormalNoise &get(Channel c) {
        if(!ready[c]) {noise[c].init(a,b,c);ready[c]=true;}
        return noise[c];
    }
    void climate(int x,int y,int z,int64_t *q) {
        const auto &shift=get(Shift);
        double px=x+4.*shift.sample(x,0,z),pz=z+4.*shift.sample(z,x,0);
        float c=get(Continentalness).sample(px,0,pz);
        finish(px,pz,y,c,q);
    }
    void finish(double px,double pz,int y,float c,int64_t *q) {
        float e=get(Erosion).sample(px,0,pz),w=get(Weirdness).sample(px,0,pz);
        float v[]={c,e,-(std::fabs(std::fabs(w)-0.6666667f)-0.33333334f)*3.f,w};
        double off=spline(0,v)+0.015f;
        float d=1.-(y*4)/128.-83./160.+off;
        float t=get(Temperature).sample(px,0,pz),h=get(Humidity).sample(px,0,pz);
        float values[]={t,h,c,e,d,w};
        for(int i=0;i<6;i++) q[i]=int64_t(10000.f*values[i]);
    }
    int biome(int x,int y,int z) { int64_t q[6];climate(x,y,z,q);return classify(q); }
};
}
