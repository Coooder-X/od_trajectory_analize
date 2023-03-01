<template>
  <div class="map-view">
    <map-comp></map-comp>
  </div>
</template>

<script>
import MapComp from '../MapComp'
export default {
  components: { MapComp },
  name: 'MapView',
  props: {
    
  }
}
</script>

<style scoped>
.map-view {
  height: 500px;
  width: 800px;
  background-color: white;
}
</style>
