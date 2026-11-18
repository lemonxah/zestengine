//! Two-dimensional panel and seam authoring. Geometry validation stays in core.
use egui::{Ui,Color32,Stroke,Pos2,Rect,Response,PointerButton,DragValue};
use glam::Vec2;
use zestengine_core::{ClothPanel,ClothPanelEdge};

#[derive(Clone)]
struct View{center:Vec2,height:f32}
impl Default for View{fn default()->Self{Self{center:Vec2::splat(0.5),height:1.4}}}
impl View{
    fn fit(points:&[Vec2])->Self{if points.is_empty(){return Self::default();}let lo=points.iter().copied().fold(Vec2::splat(f32::INFINITY),Vec2::min);let hi=points.iter().copied().fold(Vec2::splat(f32::NEG_INFINITY),Vec2::max);Self{center:(lo+hi)*0.5,height:(hi-lo).max_element().max(0.05)*1.3}}
}
struct Mapping{rect:Rect,center:Vec2,scale:f32}
impl Mapping{
    fn screen(&self,p:Vec2)->Pos2{let d=(p-self.center)*self.scale;self.rect.center()+egui::vec2(d.x,-d.y)}
    fn point(&self,p:Pos2)->Vec2{let d=p-self.rect.center();self.center+Vec2::new(d.x,-d.y)/self.scale}
    fn nearest(&self,p:Pos2,points:&[Vec2],allowed:impl Fn(usize)->bool)->Option<usize>{
        points.iter().enumerate().filter(|(i,_)|allowed(*i)).map(|(i,&q)|(i,self.screen(q).distance_sq(p))).filter(|(_,d)|*d<=100.0).min_by(|a,b|a.1.total_cmp(&b.1)).map(|(i,_)|i)
    }
}
fn canvas(ui:&mut Ui,view:&mut View,points:&[Vec2],indices:&[u32],boundary:&[Vec<u32>],highlight:&[u32],selected:Option<usize>)->(Response,Mapping){
    ui.horizontal(|ui|{if ui.small_button("Fit").clicked(){*view=View::fit(points);}ui.label("View height (m)");ui.add(DragValue::new(&mut view.height).speed(0.02).range(0.01..=1000.0));});
    let(rect,response)=ui.allocate_exact_size(egui::vec2(ui.available_width().max(100.0),240.0),egui::Sense::click_and_drag());
    let scale=rect.height()/view.height;
    if response.dragged_by(PointerButton::Middle){let d=ui.input(|i|i.pointer.delta());view.center-=Vec2::new(d.x,-d.y)/scale;}
    let map=Mapping{rect,center:view.center,scale};let painter=ui.painter_at(rect);painter.rect_filled(rect,3.0,ui.visuals().extreme_bg_color);
    let step=10.0f32.powf((view.height/6.0).log10().floor());let lo=map.point(rect.left_bottom());let hi=map.point(rect.right_top());
    for axis in 0..2{let first=(lo[axis]/step).floor()as i32;let last=(hi[axis]/step).ceil()as i32;
        for k in first..=last.min(first.saturating_add(150)){let value=k as f32*step;let(a,b)=if axis==0{(Vec2::new(value,lo.y),Vec2::new(value,hi.y))}else{(Vec2::new(lo.x,value),Vec2::new(hi.x,value))};painter.line_segment([map.screen(a),map.screen(b)],Stroke::new(0.5_f32,ui.visuals().widgets.noninteractive.bg_stroke.color));}
    }
    let mut mesh=egui::Mesh::default();let color=ui.visuals().selection.bg_fill.gamma_multiply(0.35);
    mesh.vertices.extend(points.iter().map(|&p|egui::epaint::Vertex{pos:map.screen(p),uv:egui::epaint::WHITE_UV,color}));
    mesh.indices.extend(indices.iter().copied().filter(|&i|(i as usize)<points.len()));if mesh.indices.len()%3==0{painter.add(egui::Shape::mesh(mesh));}
    if indices.len()<=30_000{for t in indices.as_chunks::<3>().0{if t.iter().any(|&i|i as usize>=points.len()){continue;}for k in 0..3{painter.line_segment([map.screen(points[t[k]as usize]),map.screen(points[t[(k+1)%3]as usize])],Stroke::new(0.5_f32,Color32::from_gray(80)));}}}
    for chain in boundary{for i in 0..chain.len(){let(a,b)=(chain[i]as usize,chain[(i+1)%chain.len()]as usize);if a<points.len()&&b<points.len(){painter.line_segment([map.screen(points[a]),map.screen(points[b])],Stroke::new(1.5_f32,ui.visuals().text_color()));}}}
    for edge in highlight.windows(2){if edge.iter().all(|&i|(i as usize)<points.len()){painter.line_segment([map.screen(points[edge[0]as usize]),map.screen(points[edge[1]as usize])],Stroke::new(3.0_f32,Color32::LIGHT_BLUE));}}
    if points.len()<=2048{for (i,&p)in points.iter().enumerate(){painter.circle_filled(map.screen(p),if selected==Some(i){4.0}else{2.0},if selected==Some(i){Color32::YELLOW}else{ui.visuals().text_color()});}}
    (response,map)
}
#[derive(Clone)]
struct Edit{topology:u64,view:View,selected:Option<usize>,error:String,undo:Vec<(usize,[f32;2],[f32;2])>,redo:Vec<(usize,[f32;2],[f32;2])>,drag:Option<(usize,[f32;2])>}
impl Edit{
    fn remember(&mut self,i:usize,before:[f32;2],after:[f32;2]){if before!=after{self.undo.push((i,before,after));if self.undo.len()>128{self.undo.remove(0);}self.redo.clear();}}
}
pub(super) fn panel_ui(ui:&mut Ui,panel:&mut ClothPanel){
    let Ok((points,indices))=panel.mesh()else{return;};let boundary=panel.boundary_loops().unwrap_or_default();let id=ui.make_persistent_id("panel-canvas");
    let topology=zestengine_core::topology_hash(points.len(),&indices);
    let mut state=ui.data(|d|d.get_temp::<Edit>(id)).filter(|s|s.topology==topology).unwrap_or(Edit{topology,view:View::fit(&points),selected:None,error:String::new(),undo:Vec::new(),redo:Vec::new(),drag:None});
    let editable=!panel.vertices.is_empty();ui.label(if editable{"Drag a vertex to reshape the panel. Middle-drag to pan."}else{"Convert to editable vertices to reshape this panel. Middle-drag to pan."});
    // Reloads or topology changes must not apply a stale edit to a new vertex.
    if state.drag.is_none()&&state.undo.last().is_some_and(|&(i,_,after)|panel.vertices.get(i)!=Some(&after)){state.undo.clear();state.redo.clear();}
    ui.horizontal(|ui|{
        if ui.add_enabled(!state.undo.is_empty(),egui::Button::new("Undo shape edit")).clicked()&& let Some(&(i,before,after))=state.undo.last(){match panel.move_vertex(i,before){Ok(())=>{state.undo.pop();state.redo.push((i,before,after));state.error.clear();},Err(e)=>state.error=e}}
        if ui.add_enabled(!state.redo.is_empty(),egui::Button::new("Redo shape edit")).clicked()&& let Some(&(i,before,after))=state.redo.last(){if panel.vertices.get(i)==Some(&before){match panel.move_vertex(i,after){Ok(())=>{state.redo.pop();state.undo.push((i,before,after));state.error.clear();},Err(e)=>state.error=e}}else{state.redo.clear();}}
    });
    let(response,map)=canvas(ui,&mut state.view,&points,&indices,&boundary,&[],state.selected);
    if (response.clicked()||response.drag_started_by(PointerButton::Primary))&& let Some(pointer)=ui.input(|i|i.pointer.press_origin()).or_else(||response.interact_pointer_pos()){state.selected=map.nearest(pointer,&points,|_|true);}
    if editable&&response.drag_started_by(PointerButton::Primary){state.drag=state.selected.and_then(|i|panel.vertices.get(i).copied().map(|p|(i,p)));}
    if editable&&response.dragged_by(PointerButton::Primary)&& let(Some(i),Some(pointer))=(state.selected,response.interact_pointer_pos()){
        state.error=panel.move_vertex(i,map.point(pointer).to_array()).err().unwrap_or_default();
    }
    if response.drag_stopped()&& let Some((i,before))=state.drag.take()&& let Some(&after)=panel.vertices.get(i){state.remember(i,before,after);}
    if let Some(i)=state.selected.filter(|&i|i<points.len()){
        ui.horizontal(|ui|{ui.label(format!("Vertex {i} (m)"));let mut p=panel.vertices.get(i).copied().unwrap_or(points[i].to_array());
            let mut changed=false;ui.add_enabled_ui(editable,|ui|{changed|=ui.add(DragValue::new(&mut p[0]).speed(0.001).prefix("X ")).changed();changed|=ui.add(DragValue::new(&mut p[1]).speed(0.001).prefix("Y ")).changed();});
            if changed{let before=panel.vertices[i];state.error=panel.move_vertex(i,p).err().unwrap_or_default();if state.error.is_empty(){state.remember(i,before,p);}}
        });
    }
    if !state.error.is_empty(){ui.colored_label(crate::theme::STATUS_WARN_FG,&state.error);}
    ui.data_mut(|d|d.insert_temp(id,state));
}
#[derive(Clone,Default)]
struct Outline{view:View,points:Vec<[f32;2]>,subdivisions:u32,error:String}
pub(super) fn outline_ui(ui:&mut Ui)->Option<ClothPanel>{
    let id=ui.make_persistent_id("outline-canvas");let mut state=ui.data(|d|d.get_temp::<Outline>(id)).unwrap_or(Outline{subdivisions:3,..Default::default()});let mut result=None;
    ui.label("Click around the panel perimeter, in order. Concave outlines are supported; edges must not cross.");
    let points=state.points.iter().copied().map(Vec2::from).collect::<Vec<_>>();let boundary=if points.len()>1{vec![(0..points.len()as u32).collect()]}else{Vec::new()};
    let(response,map)=canvas(ui,&mut state.view,&points,&[],&boundary,&[],None);
    if response.clicked()&& let Some(p)=response.interact_pointer_pos(){state.points.push(map.point(p).to_array());}
    ui.horizontal(|ui|{if ui.button("Undo point").clicked(){state.points.pop();}
        if ui.button("Clear outline").clicked(){state.points.clear();state.error.clear();}ui.label("Subdivisions");ui.add(DragValue::new(&mut state.subdivisions).range(0..=6));});
    if ui.add_enabled(state.points.len()>=3,egui::Button::new("Create panel")).clicked(){match ClothPanel::from_outline(&state.points,state.subdivisions){Ok(panel)=>{result=Some(panel);state.points.clear();state.error.clear();},Err(e)=>state.error=e}}
    if !state.error.is_empty(){ui.colored_label(crate::theme::STATUS_WARN_FG,&state.error);}
    ui.data_mut(|d|d.insert_temp(id,state));result
}
#[derive(Clone)]
struct Boundary{topology:u64,current:Vec<u32>,view:View,start:Option<u32>,end:Option<u32>,alternate:bool,error:String,panel:usize}
pub(super) fn seam_ui(ui:&mut Ui,panel:&ClothPanel,edge:&mut ClothPanelEdge){
    let Ok((points,indices))=panel.mesh()else{return;};let boundary=match panel.boundary_loops(){Ok(b)=>b,Err(e)=>{ui.label(e);return;}};
    let topology=zestengine_core::topology_hash(points.len(),&indices);
    let id=ui.make_persistent_id("seam-canvas");let mut state=ui.data(|d|d.get_temp::<Boundary>(id)).filter(|s|s.panel==edge.panel&&s.topology==topology&&s.current==edge.vertices).unwrap_or(Boundary{topology,current:edge.vertices.clone(),view:View::fit(&points),start:edge.vertices.first().copied(),end:edge.vertices.last().copied(),alternate:false,error:String::new(),panel:edge.panel});
    ui.label("Click a boundary vertex for the start, then another for the end. Blue marks the selected seam.");
    let(response,map)=canvas(ui,&mut state.view,&points,&indices,&boundary,&edge.vertices,state.start.map(|i|i as usize));let mut changed=false;
    if response.clicked()&& let Some(p)=response.interact_pointer_pos()&& let Some(i)=map.nearest(p,&points,|i|boundary.iter().any(|b|b.contains(&(i as u32)))){
        if state.start.is_none()||state.end.is_some(){state.start=Some(i as u32);state.end=None;}else{state.end=Some(i as u32);changed=true;}
    }
    changed|=ui.checkbox(&mut state.alternate,"Use the other boundary arc").changed();
    if changed&& let(Some(a),Some(b))=(state.start,state.end){match panel.boundary_path(a,b,state.alternate){Ok(chain)=>{edge.vertices=chain;state.error.clear();},Err(e)=>state.error=e}}
    if !state.error.is_empty(){ui.colored_label(crate::theme::STATUS_WARN_FG,&state.error);}
    state.current=edge.vertices.clone();ui.data_mut(|d|d.insert_temp(id,state));
}

#[cfg(test)]
mod tests{
    use super::*;
    fn frame(ctx:&egui::Context,events:Vec<egui::Event>,mut draw:impl FnMut(&mut Ui))->egui::FullOutput{
        ctx.run(egui::RawInput{screen_rect:Some(Rect::from_min_size(Pos2::ZERO,egui::vec2(600.0,600.0))),events,..Default::default()},|ctx|{egui::CentralPanel::default().show(ctx,|ui|draw(ui));})
    }
    fn canvas_rect(output:&egui::FullOutput)->Rect{
        output.shapes.iter().find_map(|s|if let egui::Shape::Rect(r)=&s.shape{((r.rect.height()-240.0).abs()<0.1&&r.rect.width()>100.0).then_some(r.rect)}else{None}).expect("panel canvas was drawn")
    }
    fn press(p:Pos2,down:bool)->Vec<egui::Event>{vec![egui::Event::PointerMoved(p),egui::Event::PointerButton{pos:p,button:PointerButton::Primary,pressed:down,modifiers:Default::default()}]}
    fn button(output:&egui::FullOutput,label:&str)->Pos2{output.shapes.iter().find_map(|s|if let egui::Shape::Text(t)=&s.shape{(t.galley.text()==label).then_some(t.pos+t.galley.size()*0.5)}else{None}).expect("button was drawn")}
    #[test]
    fn cloth_panel_canvas_drag_changes_geometry_and_rejects_inversion(){
        let ctx=egui::Context::default();let mut panel=ClothPanel::from_outline(&[[0.0,0.0],[1.0,0.0],[0.0,1.0]],0).unwrap();
        let output=frame(&ctx,vec![],|ui|panel_ui(ui,&mut panel));let view=View::fit(&panel.mesh().unwrap().0);let rect=canvas_rect(&output);let map=Mapping{rect,center:view.center,scale:rect.height()/view.height};
        let start=map.screen(Vec2::ZERO);let end=map.screen(Vec2::new(-0.1,0.0));
        frame(&ctx,press(start,true),|ui|panel_ui(ui,&mut panel));frame(&ctx,vec![egui::Event::PointerMoved(end)],|ui|panel_ui(ui,&mut panel));frame(&ctx,press(end,false),|ui|panel_ui(ui,&mut panel));
        assert!((panel.vertices[0][0]+0.1).abs()<1e-4,"dragging the displayed point must edit its panel vertex");
        let before=panel.vertices.clone();let invalid=map.screen(Vec2::splat(0.8));
        frame(&ctx,press(end,true),|ui|panel_ui(ui,&mut panel));frame(&ctx,vec![egui::Event::PointerMoved(invalid)],|ui|panel_ui(ui,&mut panel));let output=frame(&ctx,press(invalid,false),|ui|panel_ui(ui,&mut panel));
        assert_eq!(panel.vertices,before,"an inverted edit is rejected through the actual UI");
        let undo=button(&output,"Undo shape edit");frame(&ctx,press(undo,true),|ui|panel_ui(ui,&mut panel));let output=frame(&ctx,press(undo,false),|ui|panel_ui(ui,&mut panel));assert_eq!(panel.vertices[0],[0.0,0.0]);
        let redo=button(&output,"Redo shape edit");frame(&ctx,press(redo,true),|ui|panel_ui(ui,&mut panel));frame(&ctx,press(redo,false),|ui|panel_ui(ui,&mut panel));assert_eq!(panel.vertices,before);
    }
    #[test]
    fn cloth_outline_canvas_creates_a_panel_and_selects_a_sewable_boundary(){
        let ctx=egui::Context::default();let mut created=None;
        let output=frame(&ctx,vec![],|ui|{created=outline_ui(ui);});let rect=canvas_rect(&output);let view=View::default();let map=Mapping{rect,center:view.center,scale:rect.height()/view.height};
        let mut output=output;
        for point in [[0.1,0.1],[0.9,0.1],[0.9,0.4],[0.4,0.4],[0.4,0.9],[0.1,0.9]]{
            let p=map.screen(Vec2::from(point));frame(&ctx,press(p,true),|ui|{created=outline_ui(ui);});output=frame(&ctx,press(p,false),|ui|{created=outline_ui(ui);});
        }
        let button=output.shapes.iter().find_map(|s|if let egui::Shape::Text(t)=&s.shape{(t.galley.text()=="Create panel").then_some(t.pos+t.galley.size()*0.5)}else{None}).expect("create button");
        frame(&ctx,press(button,true),|ui|{created=outline_ui(ui);});frame(&ctx,press(button,false),|ui|{created=outline_ui(ui);});
        let panel=created.expect("outline clicks produce a usable custom panel");assert!(panel.vertices.len()>6);assert!(panel.boundary_loops().is_ok());
        let ctx=egui::Context::default();let mut edge=ClothPanelEdge::default();let output=frame(&ctx,vec![],|ui|seam_ui(ui,&panel,&mut edge));
        let rect=canvas_rect(&output);let points=panel.mesh().unwrap().0;let view=View::fit(&points);let map=Mapping{rect,center:view.center,scale:rect.height()/view.height};
        for i in [0,1]{let p=map.screen(points[i]);frame(&ctx,press(p,true),|ui|seam_ui(ui,&panel,&mut edge));frame(&ctx,press(p,false),|ui|seam_ui(ui,&panel,&mut edge));}
        assert_eq!(edge.vertices,panel.boundary_path(0,1,false).unwrap());assert!(edge.vertices.len()>2,"the seam includes subdivided boundary vertices");
    }
}
