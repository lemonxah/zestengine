// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Redistribute a fixed ray budget between equal-area cosine cells. Each cell
// still estimates its own mean radiance, so the resolve keeps equal cell
// weights; the proposal changes variance, never expected irradiance.
void flux_allocate_rays(vec4 lighting[4],uint uniform_count,float rotation,out uvec4 counts[4]) {
    float mean=0.0;
    for(uint group=0u;group<4u;++group) mean+=dot(lighting[group],vec4(1.0/16.0));
    vec4 weights[4];float total=0.0;
    for(uint group=0u;group<4u;++group) {
        // Bound the guide's influence. Dark/newly lit directions retain work,
        // and an old bright sample cannot consume an unbounded serial loop.
        weights[group]=vec4(1.0)+3.0*clamp(lighting[group]/max(4.0*mean,1e-8),vec4(0.0),vec4(1.0));
        total+=dot(weights[group],vec4(1.0));
    }
    uint base=max(1u,uniform_count/2u);
    uint extra=16u*(uniform_count-base),previous=0u;
    float prefix=0.0;
    for(uint bin=0u;bin<16u;++bin) {
        prefix+=weights[bin/4u][bin%4u];
        uint next=bin==15u ? extra : uint(floor(float(extra)*(prefix/total)+rotation));
        counts[bin/4u][bin%4u]=base+next-previous;
        previous=next;
    }
}
