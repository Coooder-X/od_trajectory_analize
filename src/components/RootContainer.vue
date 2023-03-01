<template>
  <div class="root-container" style="border: 2px black;">
    <div class="main">
      
    </div>
  </div>
</template>

<script>
export default {
  name: 'RootContainer',
  props: {
    
  }
}
</script>

<style scoped>
.root-container {
  position: absolute;
  height: 950px;
  width: 1900px;
}

.main {
  height: 100%;
  width: 100%;
  background-color: rgb(255, 255, 255);
  border-width: 2px;
  border-color: rgb(139 139 139);
  border-style: solid;
}
</style>
