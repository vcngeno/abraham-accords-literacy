        with col2:
            high = len(df_priority[df_priority['Priority'] == 'HIGH'])
            st.markdown(f'<div class="card" style="text-align: center;"><div class="priority-badge priority-high">High</div><div class="stat-large" style="margin-top: 16px;">{high}</div></div>', unsafe_allow_html=True)
        with col3:
            medium = len(df_priority[df_priority['Priority'] == 'MEDIUM'])
            st.markdown(f'<div class="card" style="text-align: center;"><div class="priority-badge priority-medium">Medium</div><div class="stat-large" style="margin-top: 16px;">{medium}</div></div>', unsafe_allow_html=True)
        with col4:
            low = len(df_priority[df_priority['Priority'] == 'LOW'])
            st.markdown(f'<div class="card" style="text-align: center;"><div class="priority-badge priority-low">Low</div><div class="stat-large" style="margin-top: 16px;">{low}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<h2>Top Priority Interventions</h2>', unsafe_allow_html=True)
        st.dataframe(df_priority.head(20), use_container_width=True, hide_index=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        viz_data = df_priority.groupby(['Country', 'Priority']).size().reset_index(name='Count')
        fig_priority = px.bar(viz_data, x='Country', y='Count', color='Priority', title="Priority Distribution by Country", color_discrete_map={'CRITICAL': '#FF416C', 'HIGH': '#FF8C00', 'MEDIUM': '#FFD700', 'LOW': '#32CD32'}, barmode='stack')
        fig_priority.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,46,0.4)', font={'color': '#FFFFFF', 'family': 'Inter'}, title={'font': {'size': 22, 'color': '#4ECDC4'}, 'x': 0.5, 'xanchor': 'center'}, height=500)
        st.plotly_chart(fig_priority, use_container_width=True)
    else:
        st.success("All countries meeting or exceeding SDG 4 targets.")
