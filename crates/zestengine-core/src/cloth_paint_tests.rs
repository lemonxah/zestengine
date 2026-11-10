use super::*;
#[test]
fn cloth_maps_roundtrip_remap_equal_count_reorder_and_remove_independently(){
    let p=[Vec3::ZERO,Vec3::X,Vec3::Y];let idx=[0,1,2];let mut paint=ClothPaint::seed(&p,&idx,&[1.0,0.0,0.0]);
    for (i,c) in PaintChannel::MATERIAL.into_iter().enumerate(){paint.set_values(c,&[0.0,(i+1) as f32/8.0,1.0]);}
    let text=ron::to_string(&paint).unwrap();let mut decoded:ClothPaint=ron::from_str(&text).unwrap();assert_eq!(paint,decoded);
    assert_eq!(decoded.fit(&p,&idx),PaintFit::Same);
    let reordered=[p[2],p[0],p[1]];assert_eq!(decoded.fit(&reordered,&idx),PaintFit::Remapped);
    for c in PaintChannel::MATERIAL{let old=paint.values(c);assert_eq!(decoded.values(c),vec![old[2],old[0],old[1]]);}
    assert_eq!(decoded.pin_weights(),vec![0.0,1.0,0.0]);
    let drive=decoded.values(PaintChannel::AnimationDrive);decoded.remove_channel(PaintChannel::Warp);
    assert_eq!(decoded.values(PaintChannel::Warp),vec![1.0;3]);assert_eq!(decoded.values(PaintChannel::AnimationDrive),drive);
    // A new topology gets every independently authored channel carried over.
    let expanded=[p[0],p[1],p[2],Vec3::new(0.95,0.02,0.0)];assert_eq!(paint.fit(&expanded,&[0,3,2,3,1,2]),PaintFit::Remapped);
    for c in PaintChannel::MATERIAL{let values=paint.values(c);assert_eq!(values[1],values[3]);}
}
#[test]
fn cloth_maps_brushes_mirror_and_smoothing_keep_seams_and_other_channels(){
    let positions=[Vec3::NEG_X,Vec3::X,Vec3::Y,Vec3::X];let idx=[0,1,2,0,3,2];let weld=WeldTopology::build(&positions,&idx);
    let mut paint=ClothPaint::seed(&positions,&idx,&[1.0,0.0,0.0,0.0]);
    for c in PaintChannel::MATERIAL{
        let mut values=paint.values(c);assert_eq!(values,vec![1.0;4]);
        let dab=Dab{centers:vec![Vec3::X],radius:0.2,strength:1.0,falloff:BrushFalloff::Constant,mode:BrushMode::Erase,value:0.0};
        assert!(weld.apply(&mut values,&dab));paint.set_values(c,&values);assert_eq!(values[1],values[3]);
        paint.symmetrize(c,&positions,true,0.01);assert_eq!(paint.values(c),vec![0.0,0.0,1.0,0.0]);
        let mut values=paint.values(c);assert!(weld.apply(&mut values,&Dab{mode:BrushMode::Smooth,..dab}));assert!(values[1]>0.0);assert_eq!(values[1],values[3]);
    }
    assert_eq!(paint.pin_weights(),vec![1.0,0.0,0.0,0.0]);assert!(paint.max_distance.is_empty());
}
#[test]
fn legacy_cloth_paint_has_neutral_material_maps_and_corrupt_map_keeps_healthy_channels(){
    let p=[Vec3::ZERO,Vec3::X,Vec3::Y];let idx=[0,1,2];let mut paint=ClothPaint::seed(&p,&idx,&[1.0,0.0,0.0]);
    let old_text=ron::to_string(&paint).unwrap().replace("maps:{},","");let old:ClothPaint=ron::from_str(&old_text).unwrap();
    for c in PaintChannel::MATERIAL{assert_eq!(old.values(c),vec![1.0;3]);}
    paint.set_values(PaintChannel::Warp,&[0.0,0.0,0.0]);paint.maps.insert(PaintChannel::Bend,PackedBytes::new(vec![17]));
    paint.fit(&p,&idx);assert_eq!(paint.values(PaintChannel::Warp),vec![0.0;3]);assert_eq!(paint.values(PaintChannel::Bend),vec![1.0;3]);assert_eq!(paint.pin_weights(),vec![1.0,0.0,0.0]);
}
