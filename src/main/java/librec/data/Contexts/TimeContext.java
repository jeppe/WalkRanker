package librec.data.Contexts;

public class TimeContext extends Context implements Comparable<TimeContext> {
	public int itemid;
	public int time;

	public TimeContext() {

	}

	public TimeContext(String itemid, String time) {
		this.itemid = Integer.valueOf(itemid);
		this.time = Integer.valueOf(time);
	}

	@Override
	public int compareTo(TimeContext o) {
		// TODO Auto-generated method stub
		return this.time - o.time;
	}

}