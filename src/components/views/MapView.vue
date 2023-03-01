<template>
  <div class="map-view">
    <view-header viewId="A" title="地图视图"></view-header>
    <map-comp></map-comp>
  </div>
</template>

<script>
import { defineComponent } from "vue";
import ViewHeader from '../ViewHeader'
import MapComp from '../MapComp'

export default defineComponent({
  components: { MapComp, ViewHeader },
  name: 'MapView',
  props: {
    
  }
})
</script>

<style scoped>
.map-view {
  height: 500px;
  width: 800px;
  background-color: white;
}
</style>
