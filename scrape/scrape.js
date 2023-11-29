const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs');

const messiID = 'dc5f8d51-332b-0ab2-b4b0-c97efdc624e0';
const lebronID = '1f6592b3-ff53-d321-8dc5-6038d48c1786';
const mahomesID = '37d87523-280a-9d4a-0adb-22cfc6d3619c';
const crosbyID = '706fa356-342c-1e1e-dce4-1bdd4d70a33f';
const troutID = 'e1be67bf-3688-8bd6-9bd6-c01d0e34e119';

(async function main(){
    await scrapeESPN('messi', messiID);
    await scrapeESPN('lebron', lebronID);
    await scrapeESPN('mahomes', mahomesID);
    await scrapeESPN('crosby', crosbyID);
    await scrapeESPN('trout', troutID);
})()

async function scrapeESPN(name, id){
    let url = `https://api-app.espn.com/allsports/apis/v1/now?region=us&lang=en&contentorigin=espn&limit=50&contentcategories=${id}&offset=`
    const links = []
    const stories = []
    offset = 0
    while (true) {
        console.log('offset', offset)
        const response = await axios.get(url+offset);
        let feed = response.data.feed;
        if(feed.length == 0) break;
        feed.forEach((item) => {
            if (item.links.api && item.links.api.news){
                links.push(item.links.api.news.href);
            }
        });    
        offset += 50;    
    }   

    for(let link of links){
        console.log('link', link)
        if (!link.includes("http://now.core.api.espn.com/v1/sports/news/")) continue;
        const response = await axios.get(link);
        const story = response.data.headlines[0].story;
        stories.push(story);
    }

    const jsonData = JSON.stringify(stories, null, 2);

    fs.writeFile(`../data/${name}.json`, jsonData, (err) => {
        if (err) {
            console.error('Error writing to JSON file:', err);
        } else {
            console.log(`Stories saved to ${name}.json`);
        }
    });

    console.log(name, 'stories', stories.length)
    sum = 0
    for(let story of stories){
        len = story.split(" ").length
        console.log('story', story.length)
        sum += len
    }
    console.log('avg', sum/stories.length)
    console.log("total", sum)

}